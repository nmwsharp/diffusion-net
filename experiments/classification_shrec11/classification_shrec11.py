import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from shrec11_dataset import Shrec11MeshDataset_Simplified, Shrec11MeshDataset_Original


# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
parser.add_argument("--dataset_type", type=str, help="which variant of the dataset to use ('original', or 'simplified') default: original", default = 'original')
parser.add_argument("--split_size", type=int, help="how large of a training set per-class default: 10", default=10)
args = parser.parse_args()

# system things
device = torch.device('cuda:0')
dtype = torch.float32

# problem/dataset things
n_class = 30

# model 
input_features = args.input_features # one of ['xyz', 'hks']
k_eig = 128

# training settings
n_epoch = 200
lr = 1e-3
decay_every = 50
decay_rate = 0.5
augment_random_rotate = (input_features == 'xyz')
label_smoothing_fac = 0.2


# Important paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data", "op_cache")

if args.dataset_type == "simplified":
    dataset_path = os.path.join(base_path, "data/simplified")
elif args.dataset_type == "original":
    dataset_path = os.path.join(base_path, "data/original")
else:
    raise ValueError("Unrecognized dataset type")


# === Load datasets

# Train dataset
if args.dataset_type == "simplified":
    train_dataset = Shrec11MeshDataset_Simplified(dataset_path, split_size=args.split_size,
                                                  k_eig=k_eig, op_cache_dir=op_cache_dir)
elif args.dataset_type == "original":
    train_dataset = Shrec11MeshDataset_Original(dataset_path, split_size=args.split_size,
                                                  k_eig=k_eig, op_cache_dir=op_cache_dir)
train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

# Test dataset
if args.dataset_type == "simplified":
    test_dataset = Shrec11MeshDataset_Simplified(dataset_path, split_size=None,
                                                 k_eig=k_eig, op_cache_dir=op_cache_dir,
                                                 exclude_dict=train_dataset.entries)
elif args.dataset_type == "original":
    test_dataset = Shrec11MeshDataset_Original(dataset_path, split_size=None,
                                                 k_eig=k_eig, op_cache_dir=op_cache_dir,
                                                 exclude_dict=train_dataset.entries)
test_loader = DataLoader(test_dataset, batch_size=None)





# === Create the model

C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=n_class,
                                          C_width=64, 
                                          N_block=4, 
                                          last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                                          outputs_at='global_mean', 
                                          dropout=False)


model = model.to(device)

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
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

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
            verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        # Apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

        # Evaluate loss
        loss = diffusion_net.utils.label_smoothing_log_loss(preds, labels, label_smoothing_fac)
        loss.backward()
        
        # track accuracy
        pred_labels = torch.max(preds, dim=-1).indices
        this_correct = pred_labels.eq(labels).sum().item()
        correct += this_correct
        total_num += 1

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    train_acc = correct / total_num
    return train_acc


# Do an evaluation pass on the test dataset 
def test():
    
    model.eval()
    
    correct = 0
    total_num = 0
    with torch.no_grad():
    
        for data in tqdm(test_loader):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

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
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

            # track accuracy
            pred_labels = torch.max(preds, dim=-1).indices
            this_correct = pred_labels.eq(labels).sum().item()
            correct += this_correct
            total_num += 1

    test_acc = correct / total_num
    return test_acc 


print("Training...")

for epoch in range(n_epoch):
    train_acc = train_epoch(epoch)
    test_acc = test()
    print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.3f}%".format(epoch, 100*train_acc, 100*test_acc))

# Test
test_acc = test()
print("Overall test accuracy: {:06.3f}%".format(100*test_acc))
