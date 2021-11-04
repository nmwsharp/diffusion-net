import shutil
import os
import sys
import random
import numpy as np

import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import toNP




class RNAMeshDataset(Dataset):
    """
    RNA segmentation dataset from Poulenard et al., 3DV 2019.

    See https://github.com/nmwsharp/RNA-Surface-Segmentation-Dataset
    """

    def __init__(self, root_dir, train, k_eig, use_cache=True, op_cache_dir=None):

        self.train = train  # bool
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.n_class = 260 # (includes -1)

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-vertex 

        # Load the meshes & labels
        if self.train:
            with open(os.path.join(self.root_dir, "train.txt")) as f:
                this_files = [line.rstrip() for line in f]
        else:
            with open(os.path.join(self.root_dir, "test.txt")) as f:
                this_files = [line.rstrip() for line in f]

        print("loading {} files: {}".format(len(this_files), this_files))

        # Load the actual files

        off_path = os.path.join(root_dir, "off")
        label_path = os.path.join(root_dir, "labels")
        for f in this_files:
            off_file = os.path.join(off_path, f)
            label_file = os.path.join(label_path, f[:-4] + ".txt")

            verts, faces = pp3d.read_mesh(off_file)
            labels = np.loadtxt(label_file).astype(int) + 1 # shift -1 --> 0
        
            verts = torch.tensor(verts).float()
            faces = torch.tensor(faces)
            labels = torch.tensor(labels)

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.labels_list.append(labels)

        # Precompute operators
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

    def __len__(self):
        return len(self.verts_list)
    
    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.labels_list[idx]
