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

class Shrec11MeshDataset_Original(Dataset):

    # NOTE: Original data from the challenge, not simplified models
  
    # The original SHREC11 models were previously distributed via NIST [here](https://www.nist.gov/itl/iad/shrec-2011-datasets), but that page seems to have been lost to the sands of time. We provide a zip of the old dataset page here: https://drive.google.com/uc?export=download&id=1O_P03aAxhjCOKQH2n71j013-EfSmEp5e. The relevant files are in `SHREC11_test_database_new.zip`, which is password protected with the password `SHREC11@NIST`. 

    # Unzip it like
    # unzip -P SHREC11@NIST SHREC11_test_database_new.zip -d [DATA_ROOT]/raw
    
    def __init__(self, root_dir, split_size, k_eig, exclude_dict=None, op_cache_dir=None):
        
        self.root_dir = root_dir
        self.n_class = 30 
        self.split_size = split_size # pass None to take all entries (except those in exclude_dict)
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir

        self.class_names = []
        self.entries = {}

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []

        raw_path = os.path.join(self.root_dir, 'raw')

        
        ## Parse the categories file
        cat_path = os.path.join(self.root_dir, 'categories.txt')
        with open(cat_path) as cat_file:
            cat_file.readline() # skip the first two lines
            cat_file.readline() 
            for i_class in range(30):
                cat_file.readline() 
                class_name, _, count = cat_file.readline().strip().split()
                count = int(count)
                self.class_names.append(class_name)
                mesh_list = []
                for j in range(20):
                    mesh_list.append(cat_file.readline().strip())


                # Randomly grab samples for this split. If given, disallow any samples in commmon with exclude_dict (ie making sure train set is distinct from test).
                order = np.random.permutation(len(mesh_list))
                added = 0
                self.entries[class_name] = set()
                for ind in order:
                    if(split_size is not None and added == split_size): continue

                    name = mesh_list[ind]
                    if exclude_dict is not None and name in exclude_dict[class_name]:
                        continue

                    path = os.path.join(root_dir, "raw", "T{}.off".format(name))

                    verts, faces = pp3d.read_mesh(path)
                    verts = torch.tensor(verts).float()
                    faces = torch.tensor(faces)

                    # center and unit scale
                    verts = diffusion_net.geometry.normalize_positions(verts)

                    self.verts_list.append(verts)
                    self.faces_list.append(faces)
                    self.labels_list.append(i_class)
                    self.entries[class_name].add(name)

                    added += 1

                print(class_name + " -- " + " ".join([p for p in self.entries[class_name]]))

                if(split_size is not None and added < split_size):
                    raise ValueError("could not find enough entries to generate requested split")
                
        for ind, label in enumerate(self.labels_list):
            self.labels_list[ind] = torch.tensor(label)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        verts = self.verts_list[idx]
        faces = self.faces_list[idx]
        label = self.labels_list[idx]
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
        return verts, faces, frames, mass, L, evals, evecs, gradX, gradY, label

class Shrec11MeshDataset_Simplified(Dataset):

    # NOTE: Remeshed data from MeshCNN authors.
    # Can be downloaded here (link from the MeshCNN authors): https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz. Note that despite the filename, this really is the shapes from the SHREC 2011 dataset. Extract it to the `[ROOT_DIR]/raw/` directory.

    def __init__(self, root_dir, split_size, k_eig, exclude_dict=None, op_cache_dir=None):
        
        self.root_dir = root_dir
        self.n_class = 30 
        self.split_size = split_size # pass None to take all entries (except those in exclude_dict)
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir

        self.class_names = [ 'alien', 'ants', 'armadillo', 'bird1', 'bird2', 'camel', 'cat', 'centaur', 'dinosaur', 'dino_ske', 'dog1', 'dog2', 'flamingo', 'glasses', 'gorilla', 'hand', 'horse', 'lamp', 'laptop', 'man', 'myScissor', 'octopus', 'pliers', 'rabbit', 'santa', 'shark', 'snake', 'spiders', 'two_balls', 'woman']
        
        self.entries = {}

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []

        raw_path = os.path.join(self.root_dir, 'raw', "shrec_16")

        for class_idx, class_name in enumerate(self.class_names):
            
            # load both train and test subdirectories; we are manually regenerating random splits to do multiple trials
            mesh_files = []
            for t in ['test', 'train']:
                files = os.listdir(os.path.join(raw_path, class_name, t))
                for f in files:
                    full_f = os.path.join(raw_path, class_name, t, f)
                    mesh_files.append(full_f)


            # Randomly grab samples for this split. If given, disallow any samples in commmon with exclude_dict (ie making sure train set is distinct from test).
            order = np.random.permutation(len(mesh_files))
            added = 0
            self.entries[class_name] = set()
            for ind in order:
                if(split_size is not None and added == split_size): continue

                path = mesh_files[ind]
                if exclude_dict is not None and path in exclude_dict[class_name]:
                    continue

                verts, faces = pp3d.read_mesh(path)
                verts = torch.tensor(verts).float()
                faces = torch.tensor(faces)

                # center and unit scale
                verts = diffusion_net.geometry.normalize_positions(verts)

                self.verts_list.append(verts)
                self.faces_list.append(faces)
                self.labels_list.append(class_idx)
                self.entries[class_name].add(path)

                added += 1

            print(class_name + " -- " + " ".join([os.path.basename(p) for p in self.entries[class_name]]))

            if(split_size is not None and added < split_size):
                raise ValueError("could not find enough entries to generate requested split")
            
        for ind, label in enumerate(self.labels_list):
            self.labels_list[ind] = torch.tensor(label)

        # Precompute operators
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.labels_list[idx]

