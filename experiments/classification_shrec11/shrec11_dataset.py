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

class Shrec11MeshDataset_Simplified(Dataset):
    # NOTE: Remeshed data from MeshCNN authors.

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
            
        for ind, labels in enumerate(self.labels_list):
            self.labels_list[ind] = torch.tensor(labels)

        # Precompute operators
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.labels_list[idx]

class Shrec11Dataset(Dataset):
    """SHREC11 classification dataset"""

    def __init__(self, root_dir, train, k_eig=128, use_cache=True, op_cache_dir=None):

        self.train = train  # bool
        self.k_eig = k_eig 
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.n_class = 8

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-face labels!!

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            load_cache = train_cache if self.train else test_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list = torch.load( load_cache)
                return
            print("  --> dataset not in cache, repopulating")


        # Load the meshes & labels

        # Get all the files
        mesh_files = []
        label_files = []

        # Train test split
        if self.train:
    
            # adobe
            mesh_dirpath = os.path.join(self.root_dir, "meshes", "train", "adobe")
            label_dirpath = os.path.join(self.root_dir, "segs", "train", "adobe")
            for fname in os.listdir(mesh_dirpath):
                mesh_fullpath = os.path.join(mesh_dirpath, fname)
                label_fullpath = os.path.join(label_dirpath, fname[:-4] + ".txt")
                mesh_files.append(mesh_fullpath)
                label_files.append(label_fullpath)
            
            # faust
            mesh_dirpath = os.path.join(self.root_dir, "meshes", "train", "faust")
            label_dirpath = os.path.join(self.root_dir, "segs", "train", "faust")
            for fname in os.listdir(mesh_dirpath):
                mesh_fullpath = os.path.join(mesh_dirpath, fname)
                label_fullpath = os.path.join(label_dirpath, "faust_corrected.txt")
                mesh_files.append(mesh_fullpath)
                label_files.append(label_fullpath)
            
            # mit
            mesh_dirpath_patt = os.path.join(self.root_dir, "meshes", "train", "MIT_animation", "meshes_{}", "meshes")
            label_dirpath = os.path.join(self.root_dir, "segs", "train", "mit")
            pose_names = ['bouncing','handstand','march1','squat1', 'crane','jumping', 'march2', 'squat2']
            for pose in pose_names:
                mesh_dirpath = mesh_dirpath_patt.format(pose)
                for fname in os.listdir(mesh_dirpath):
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    label_fullpath = os.path.join(label_dirpath, "mit_{}_corrected.txt".format(pose))
                    mesh_files.append(mesh_fullpath)
                    label_files.append(label_fullpath)
            
            # scape
            mesh_dirpath = os.path.join(self.root_dir, "meshes", "train", "scape")
            label_dirpath = os.path.join(self.root_dir, "segs", "train", "scape")
            for fname in os.listdir(mesh_dirpath):
                mesh_fullpath = os.path.join(mesh_dirpath, fname)
                label_fullpath = os.path.join(label_dirpath, "scape_corrected.txt")
                mesh_files.append(mesh_fullpath)
                label_files.append(label_fullpath)
            
        else:

            # shrec
            mesh_dirpath = os.path.join(self.root_dir, "meshes", "test", "shrec")
            label_dirpath = os.path.join(self.root_dir, "segs", "test", "shrec")
            for iShrec in range(1,21):
                if iShrec == 16 or iShrec == 18: continue # why are these messing from the dataset? so many questions...
                if iShrec == 12:
                    mesh_fname = "12_fix_orientation.off"
                else:
                    mesh_fname = "{}.off".format(iShrec)
                label_fname = "shrec_{}_full.txt".format(iShrec)
                mesh_fullpath = os.path.join(mesh_dirpath, mesh_fname)
                label_fullpath = os.path.join(label_dirpath, label_fname)
                mesh_files.append(mesh_fullpath)
                label_files.append(label_fullpath)

        print("loading {} meshes".format(len(mesh_files)))

        # Load the actual files
        for iFile in range(len(mesh_files)):

            print("loading mesh " + str(mesh_files[iFile]))

            verts, faces = pp3d.read_mesh(mesh_files[iFile])
            labels = np.loadtxt(label_files[iFile]).astype(int)-1

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            labels = torch.tensor(np.ascontiguousarray(labels))

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.labels_list.append(labels)

        for ind, labels in enumerate(self.labels_list):
            self.labels_list[ind] = labels

        # Precompute operators
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        # save to cache
        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list), load_cache)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.labels_list[idx]
