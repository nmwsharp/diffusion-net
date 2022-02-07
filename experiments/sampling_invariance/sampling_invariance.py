import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np



sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from faust_with_robust_test_dataset import FaustWithRobustTestDataset


# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: xyz", default = 'xyz')
args = parser.parse_args()


# system things
device = torch.device('cuda:0')
dtype = torch.float32

# problem/dataset things
n_class = 6890

# model 
input_features = args.input_features # one of ['xyz', 'hks']
k_eig = 128

# training settings
train = not args.evaluate
n_epoch = 200
lr = 1e-3
decay_every = 50
decay_rate = 0.5
augment_random_rotate = (input_features == 'xyz')



# Important paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data", "op_cache")
geodesic_cache_dir = os.path.join(base_path, "data", "geodesic_cache") # for evaluating error metrics
pretrain_path = os.path.join(base_path, "pretrained_models/categorical_correspondence_{}_4x256.pth".format(input_features))
model_save_path = os.path.join(base_path, "data/saved_models/categorical_correspondence_{}_4x256.pth".format(input_features))
dataset_path = os.path.join(base_path, "data")


# === Load datasets

# Load the test dataset
test_dataset = FaustWithRobustTestDataset(dataset_path, train=False, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)

# Load the train dataset
train_dataset = FaustWithRobustTestDataset(dataset_path, train=True, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

# Use the first shape in the training dataset as the reference model (used for measuring geodesic error below)
verts_ref, faces_ref = train_dataset[0][:2]
verts_ref.requires_grad = False
faces_ref.requires_grad = False


# === Create the model

C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=n_class,
                                          C_width=256, 
                                          N_block=4, 
                                          last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                                          outputs_at='vertices', 
                                          dropout=True)


model = model.to(device)

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))
    print("...done")


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
    
    correct = 0
    total_num = 0
    for data in tqdm(train_loader):

        # Get data
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, _ = data

        # Move to device
        verts = verts.to(device)
        faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        labels = labels.to(device)
        
        # Randomly rotate positions
        if augment_random_rotate:
            # Rotate about up (Y-axis) only
            verts = diffusion_net.utils.random_rotate_points_y(verts)

        # Construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        # Apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

        # Evaluate loss
        loss = torch.nn.functional.nll_loss(preds, labels)
        loss.backward()
        
        # track accuracy
        pred_labels = torch.max(preds, dim=1).indices
        this_correct = pred_labels.eq(labels).sum().item()
        this_num = labels.shape[0]
        correct += this_correct
        total_num += this_num

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    train_acc = correct / total_num
    return train_acc


# Do an evaluation pass on the test dataset 
def test(with_geodesic_error=False):

    if with_geodesic_error:
        print("Evaluating geodesic error metrics")
    
    model.eval()
    
    correct = 0
    total_num = 0
    with torch.no_grad():

        # If measuring geodesic error, keep track of it (for each type of mutation)
        mut_geodesic_errors = {}
   
        i = 0
        for data in tqdm(test_loader):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, mut = data

               
            # Move to device
            verts = verts.to(device)
            faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)
            
            # Construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

            # Apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

            # track accuracy
            pred_labels = torch.max(preds, dim=1).indices
            this_correct = pred_labels.eq(labels).sum().item()
            this_num = labels.shape[0]
            correct += this_correct
            total_num += this_num

            if with_geodesic_error:

                pred_labels = torch.max(preds, dim=-1).indices
                errors = diffusion_net.geometry.geodesic_label_errors(verts_ref, faces_ref, pred_labels, labels, normalization='diameter', geodesic_cache_dir=geodesic_cache_dir)

                if mut not in mut_geodesic_errors:
                    mut_geodesic_errors[mut] = []
                mut_geodesic_errors[mut].extend(errors)


        if with_geodesic_error:
            print("\n== Geodesic errors ==")
            for mut in mut_geodesic_errors:
                print("    {:>8}  mean: {:.2f}".format(mut, 100*np.mean(mut_geodesic_errors[mut])))

    test_acc = correct / total_num
    return test_acc 


if train:
    print("Training...")

    for epoch in range(n_epoch):
        train_acc = train_epoch(epoch)
        test_acc = test()
        print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.3f}%".format(epoch, 100*train_acc, 100*test_acc))

    print(" ==> saving last model to " + model_save_path)
    torch.save(model.state_dict(), model_save_path)


# Test
test_acc = test(with_geodesic_error=True)
print("Overall test accuracy: {:06.3f}%".format(100*test_acc))
