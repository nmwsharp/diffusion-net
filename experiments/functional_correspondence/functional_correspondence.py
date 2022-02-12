import os
import sys
import argparse
import random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn
from torch.utils.data import DataLoader

from fmaps_model import FunctionalMapCorrespondenceWithDiffusionNetFeatures

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import toNP

from faust_scape_dataset import FaustScapeDataset

# === Options


# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--train_dataset", type=str, default="faust", help="what dataset to train on")
parser.add_argument("--test_dataset", type=str, default="faust", help="what dataset to test on")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
parser.add_argument("--load_model", type=str, help="path to load a pretrained model from")
args = parser.parse_args()

# system things
device = torch.device('cuda')
dtype = torch.float32

# model 
input_features = args.input_features # one of ['xyz', 'hks']
k_eig = 128

# functional maps settings
n_fmap = 30 # number of eigenvectors used within functional maps
n_feat = 128 # dimension of features computed by DiffusionNet extractor
lambda_param = 1e-3 # functional map block regularization parameter

# training settings
train = not args.evaluate
n_epoch = 5
lr = 5e-4
decay_every = 9999
decay_rate = 0.1
augment_random_rotate = (input_features == 'xyz')


# Important paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data", "op_cache")
geodesic_cache_dir = os.path.join(base_path, "data", "geodesic_cache") # for evaluating error metrics
model_save_path = os.path.join(base_path, "saved_models/{}_{}.pth".format(args.train_dataset, input_features))
dataset_path = os.path.join(base_path, "data")
diffusion_net.utils.ensure_dir_exists(os.path.join(base_path, "saved_models/"))


# === Load datasets

if not args.evaluate: 
    train_dataset = FaustScapeDataset(dataset_path, name=args.train_dataset, train=True, k_eig=k_eig, n_fmap=n_fmap, use_cache=True, op_cache_dir=op_cache_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True)

test_dataset = FaustScapeDataset(dataset_path, name=args.test_dataset, train=False, k_eig=k_eig, n_fmap=n_fmap, use_cache=True, op_cache_dir=op_cache_dir)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, shuffle=False)


# === Create the model

C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

model = FunctionalMapCorrespondenceWithDiffusionNetFeatures(
                        n_feat=n_feat,
                        n_fmap=n_fmap,
                        input_features=input_features, 
                        lambda_param=lambda_param
                        )


model = model.to(device)

if args.load_model:
    # load the pretrained model
    print("Loading pretrained model from: " + str(args.load_model))
    model.load_state_dict(torch.load(args.load_model))
    print("...done")

if args.evaluate and not args.load_model:
    raise ValueError("Called with --evaluate but not --load_model. This will evaluate on a randomly initialized model, which is probably not what you want to do.")


# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train_epoch(epoch):

    # Implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr 
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 


    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()

    losses = []
    
    for data in tqdm(train_loader):

        # Get data
        shape1, shape2, C_gt = data
        *shape1, name1 = shape1
        *shape2, name2 = shape2
        shape1, shape2, C_gt = [x.to(device) for x in shape1], [x.to(device) for x in shape2], C_gt.to(device).unsqueeze(0)

        # Randomly rotate positions
        if augment_random_rotate:
            shape1[0] = diffusion_net.utils.random_rotate_points(shape1[0])
            shape2[0] = diffusion_net.utils.random_rotate_points(shape2[0])

        # Apply the model
        C_pred, feat1, feat2 = model(shape1, shape2)

        # Evaluate loss 
        loss = torch.mean(torch.square(C_pred-C_gt)) # L2 loss
        losses.append(toNP(loss))
        loss.backward()

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    train_loss = np.mean(losses)

    return train_loss


# Do an evaluation pass on the test dataset 
def test(with_geodesic_error=False):

    if with_geodesic_error:
        print("Evaluating geodesic error metrics")
    
    model.eval()

    losses = []
    geodesic_errors = []
    
    with torch.no_grad():

        for data in tqdm(test_loader):
        
            # Get data
            shape1, shape2, C_gt = data
            *shape1, name1 = shape1
            *shape2, name2 = shape2 
            shape1, shape2, C_gt = [x.to(device) for x in shape1], [x.to(device) for x in shape2], C_gt.to(device)
        
            verts1_orig = shape1[0]
            if augment_random_rotate:
                shape1[0] = diffusion_net.utils.random_rotate_points(shape1[0])
                shape2[0] = diffusion_net.utils.random_rotate_points(shape2[0])

            # Apply the model
            C_pred, feat1, feat2 = model(shape1, shape2)
            C_pred = C_pred.squeeze(0)

            # Loss
            loss = torch.mean(torch.square(C_pred-C_gt)) # L2 loss
            losses.append(toNP(loss))

            # Compute the geodesic error in the vertex-to-vertex correspondence
            if with_geodesic_error:

                # gather values
                verts1 = shape1[0]
                faces1 = shape1[1]
                evec1 = shape1[6]
                vts1 = shape1[10]
                verts2 = shape2[0]
                faces2 = shape2[1]
                evec2 = shape2[6]
                vts2 = shape2[10]

                # construct a vertex-to-vertex map via nearest neighbors from the functional map
                evec1_on_2 = evec1[:,:n_fmap] @ C_pred.squeeze(0).transpose(0,1)
                _, pred_labels2to1 = diffusion_net.geometry.find_knn(evec2[:,:n_fmap], evec1_on_2, k=1, method='cpu_kd')
                pred_labels2to1 = pred_labels2to1.squeeze(-1)

                # measure the geodesic error for each template vertex along shape 1
                vts2on1 = pred_labels2to1[vts2]

                errors = diffusion_net.geometry.geodesic_label_errors(verts1_orig, faces1, vts2on1, vts1, normalization='area', geodesic_cache_dir=geodesic_cache_dir)
                
                geodesic_error = toNP(torch.mean(errors))
                geodesic_errors.append(geodesic_error)


    mean_loss = np.mean(losses)
    mean_geodesic_error = np.mean(geodesic_errors) if with_geodesic_error else -1

    return mean_loss, mean_geodesic_error

if train:
    print("Training...")

    for epoch in range(n_epoch):
        train_loss = train_epoch(epoch)
        test_loss, test_geodesic_error = test(with_geodesic_error=True)
        print("Epoch {} - Train overall: {:.5e}  Test overall: {:.5e}  Test geodesic error: {:.5e}".format(epoch, train_loss, test_loss, test_geodesic_error))

        print(" ==> saving last model to " + model_save_path)
        torch.save(model.state_dict(), model_save_path)


# Test
mean_loss, mean_geodesic_error = test(with_geodesic_error=True)
print("Overall test accuracy: {:.5e}  geodesic error: {:.5e}".format(mean_loss, mean_geodesic_error))
