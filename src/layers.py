import sys
import os
import random

import scipy
import scipy.sparse.linalg as sla
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import numpy as np
import torch
import torch.nn as nn

from utils import toNP
import geometry
import utils



class LaplacianBlock(nn.Module):
    """
    Applies Laplacian diffusion in the spectral domain like
        f_out = e ^ (lambda_i t) f_in
    with learned per-channel parameters t.

    Inputs:
      - values: (K,C) in the spectral domain
      - evals: (K) eigenvalues
    Outputs:
      - (K,C) transformed values in the spectral domain
    """

    def __init__(self, C_inout):
        super(LaplacianBlock, self).__init__()
        self.C_inout = C_inout

        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)
        nn.init.constant_(self.diffusion_time, 0.0001)

    def forward(self, x, evals):

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))

        diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * torch.abs(self.diffusion_time).unsqueeze(0)) 
        return diffusion_coefs * x


class PairwiseDot(nn.Module):
    """
    Compute dot-products between input vectors with a learned complex-linear layer.
    
    Input:
        - vectors: (V,C,2)
    Output:
        - dots: (V,C) dots 
    """

    def __init__(self, C_inout, linear_complex=True):
        super(PairwiseDot, self).__init__()

        self.C_inout = C_inout
        self.linear_complex = linear_complex

        if(self.linear_complex):
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)

    def forward(self, vectors):

        vectorsA = vectors # (V,C)

        if self.linear_complex:
            vectorsBreal = self.A_re(vectors[...,0]) - self.A_im(vectors[...,1])
            vectorsBimag = self.A_re(vectors[...,1]) + self.A_im(vectors[...,0])
        else:
            vectorsBreal = self.A(vectors[...,0])
            vectorsBimag = self.A(vectors[...,1])

        dots = vectorsA[...,0] * vectorsBreal + vectorsA[...,1] * vectorsBimag

        return torch.tanh(dots)


class MiniMLP(nn.Sequential):
    '''
    A simple MLP with configurable hidden layer sizes.
    '''
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i),
                    nn.Dropout(p=.5)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i),
                    activation()
                )


class DiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_inout, C_hidden,
                 dropout=False, pairwise_dot=True, dot_linear_complex=True):
        super(DiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.C_inout = C_inout
        self.C_hidden = C_hidden

        self.dropout = dropout
        self.pairwise_dot = pairwise_dot
        self.dot_linear_complex = dot_linear_complex

        # Laplacian block
        self.spec0 = LaplacianBlock(self.C_inout)
        
        self.C_mlp = 2*self.C_inout
      
        if self.pairwise_dot:
            self.pairwise_dot = PairwiseDot(self.C_inout, linear_complex=self.dot_linear_complex)
            self.C_mlp += self.C_inout

        # MLPs
        self.mlp0 = MiniMLP([self.C_mlp] + self.C_hidden + [self.C_inout], dropout=self.dropout)


    def forward(self, x0, mass, evals, evecs, grad_from_spectral):

        if x0.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x0.shape, self.C_inout))

        # Transform to spectral
        x0_spec = geometry.to_basis(x0, evecs, mass)
        
        # Laplacian block 
        x0_spec = self.spec0(x0_spec, evals)

        # Transform back to per-vertex 
        x0_lap = geometry.from_basis(x0_spec, evecs)
        x0_comb = torch.cat((x0, x0_lap), dim=-1)

        if self.pairwise_dot:
            # If using the pairwise dot block, add it to the scalar values as well
            x0_grad = utils.cmatvecmul_stacked(grad_from_spectral, x0_spec)
            x0_gradprods = self.pairwise_dot(x0_grad)
            x0_comb = torch.cat((x0_comb, x0_gradprods), dim=-1)
        
        # Apply the mlp
        x0_out = self.mlp0(x0_comb)

        # Skip connection
        x0_out = x0_out + x0

        return x0_out
