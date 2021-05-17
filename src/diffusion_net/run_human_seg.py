import os
import sys
import random
import time

import scipy
import scipy.sparse.linalg as sla
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import progressbar
import numpy as np
import igl

import geometry
import utils
import layers
from human_seg_dataset import ShapeSegMeshDataset
from utils import toNP


# === Experiment options

# system things
opts = utils.get_default_opts()
root_path = os.path.join(os.path.dirname(__file__))
dataset_path = os.path.join(root_path, "../data/human_seg")
opts.eigensystem_cache_dir = os.path.join(root_path, "../cache")
device = torch.device('cuda:0')
dtype = torch.float32

# parameters & training settings
opts.k_eig = 128
opts.lr = 1e-3
opts.decay_every = 50
opts.decay_rate = 0.5
batch_size = 1
augment_random_rotate = True

n_class = 8
test_dataset = ShapeSegMeshDataset(opts, dataset_path, train=False)
test_loader = DataLoader(test_dataset, batch_size=1)
train_dataset = ShapeSegMeshDataset(opts, dataset_path, train=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# === Create the model

class DiffusionNet(torch.nn.Module):
    def __init__(self):
        super(DiffusionNet, self).__init__()
        
        n_in = 3
        n_out = n_class
        n_width = 32
        n_block = 4
        self.blocks = []
        hidden_dims = [n_width, n_width]
        for i_block in range(n_block):
                
            C1_in = n_width
            C1_out = n_width

            self.blocks.append(layers.DiffusionNetBlock(C_inout=n_width, C_hidden=hidden_dims))
            self.add_module("block_"+str(i_block), self.blocks[-1])

        # First and last linear layers
        self.first_lin0 = nn.Linear(n_in, n_width)
        self.last_lin0 = nn.Linear(n_width, n_out)


    def forward(self, verts, faces, frames, mass, evals, evecs, grad_from_spectral):

        x0 = geometry.normalize_positions(verts)
        
        # Apply the first linear layer
        x0 = self.first_lin0(x0)
       
        for b in self.blocks:
            x0 = b(x0, mass, evals, evecs, grad_from_spectral)
        
        # Apply the last linear layer
        x0 = self.last_lin0(x0)
        
        # Final (log) softmax
        return F.log_softmax(x0, dim=-1)

model = DiffusionNet()
model = model.to(device)

# === Optimize

optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)

def train(epoch):

    if epoch > 0 and epoch % opts.decay_every == 0:
        opts.lr = opts.lr * opts.decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = opts.lr 


    model.train()
    optimizer.zero_grad()
    
    correct = 0
    total_num = 0

    for data in progressbar.progressbar(train_loader):
        
        verts, faces, frames, mass, evals, evecs, grad_from_spectral, labels = data
        verts = verts.to(device)
        faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        grad_from_spectral = grad_from_spectral.to(device)
        labels = labels.to(device)

        # Randomly rotate positions
        if augment_random_rotate:
            verts = utils.random_rotate_points(verts)

        preds = model(verts, faces, frames, mass, evals, evecs, grad_from_spectral)

        if torch.any(torch.isnan(preds)):
            raise ValueError("NaN outputs :(")

        loss = F.nll_loss(preds.transpose(-1,-2), labels)
        loss.backward()
        
        # track accuracy
        pred_labels = torch.max(preds[0,...], dim=1).indices
        this_correct = pred_labels.eq(labels[0,...]).sum().item()
        this_num = labels.shape[1]
        correct += this_correct
        total_num += this_num

        optimizer.step()
        optimizer.zero_grad()

    train_acc = correct / total_num
    print("Epoch {} - Train: {:06.4f}".format(epoch, train_acc))

        
def test():
    model.eval()
    correct = 0
    total_num = 0
    for i, data in enumerate(test_loader):

        verts, faces, frames, mass, evals, evecs, grad_from_spectral, labels = data
        verts = verts.to(device)
        faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        grad_from_spectral = grad_from_spectral.to(device)
        labels = labels.to(device)
            
        preds = model(verts, faces, frames, mass, evals, evecs, grad_from_spectral)
        if torch.any(torch.isnan(preds)):
            raise ValueError("NaN outputs :(")


        # track accuracy
        pred_labels = torch.max(preds[0,...], dim=1).indices
        this_correct = pred_labels.eq(labels[0,...]).sum().item()
        this_num = labels.shape[1]
        correct += this_correct
        total_num += this_num

    return correct / total_num

print("Training...")
for epoch in range(200):
    train(epoch)
    test_acc = test()
    print("Epoch {} - Test: {:06.4f}".format(epoch, test_acc))

print("last test acc: " + str(test_acc))
