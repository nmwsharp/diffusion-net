import shutil
import os
import sys
import random
import numpy as np

import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d
import plyfile

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import toNP


class FaustWithRobustTestDataset(Dataset):
    """FAUST correspondence dataset, with robust test. Target data is vertex indices on the template mesh. Training data is original FAUST models, testing data are remeshed versions according to several strategies to measure discretization invariance, from https://github.com/nmwsharp/discretization-robust-correspondence-benchmark"""

    def __init__(self, root_dir, train, k_eig=128, use_cache=True, op_cache_dir=None):

        self.train = train  # bool
        self.k_eig = k_eig 
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.n_class = 6890

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.normals_list = []
        self.labels_list = []  # per-vertex labels!!
        self.mut_list = []     # for the test dataset, which mutation is it

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            load_cache = train_cache if self.train else test_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.faces_list, self.normals_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list, self.mut_list = torch.load(load_cache)
                return
            print("  --> dataset not in cache, repopulating")


        # Load the meshes & labels

        # Get all the files
        train_dirpath_base = os.path.join(self.root_dir, "train", "MPI-FAUST", "training", "registrations")
        test_dirpath_base = os.path.join(self.root_dir, "test", "data")

        if self.train:

            # Path to training meshes

            for i in range(80):
                # load mesh
                mesh_fullpath = os.path.join(train_dirpath_base, "tr_reg_{:03d}.ply".format(i))
                verts, faces = pp3d.read_mesh(mesh_fullpath)
                normals = None

                # convert to torch
                verts = torch.tensor(np.ascontiguousarray(verts)).float()
                faces = torch.tensor(np.ascontiguousarray(faces))
                
                # center and scale
                verts = diffusion_net.geometry.normalize_positions(verts, method='bbox')
                
                # these are already template meshes, so target indices are just identity
                labels = torch.arange(verts.shape[0])

                self.verts_list.append(verts)
                self.faces_list.append(faces)
                self.normals_list.append(normals)
                self.labels_list.append(labels)
                self.mut_list.append(None)

                print("loaded {} train meshes".format(len(self.verts_list)))
        else:

            methods = ['orig', 'iso', 'qes', 'mc', 'dense', 'cloud']
            
            for method in methods:
                for i in range(80, 100):

                    if method == 'orig':
                        # Load original meshes from the "train" directory (which really just means the FAUST meshes)

                        mesh_fullpath = os.path.join(train_dirpath_base, "tr_reg_{:03d}.ply".format(i))
                        verts, faces = pp3d.read_mesh(mesh_fullpath)
                        normals = None

                        # these are already template meshes, so target indices are just identity
                        labels = torch.arange(verts.shape[0])
                    
                    elif method == 'cloud':
                        # Need to load cloud separately, because the pp3d loader complains about faces (and we want to read normals)

                        mesh_fullpath = os.path.join(test_dirpath_base, method, "tr_reg_{}_{:03d}.ply".format(method, i))
                        labels_fullpath = os.path.join(test_dirpath_base, method, "tr_reg_{}_{:03d}.txt".format(method, i))

                        # Use the open3d ply reader (booo, dependencies)
                        with open(mesh_fullpath, 'rb') as f:
                            data = plyfile.PlyData.read(f)

                        verts = np.stack(([data['vertex'][axis] for axis in ['x', 'y', 'z']]), axis=-1)
                        faces = np.zeros((0,3), dtype=np.int64)
                        normals = np.stack(([data['vertex'][axis] for axis in ['nx', 'ny', 'nz']]), axis=-1)
                        normals = torch.tensor(np.ascontiguousarray(normals)).float()

                        labels = torch.from_numpy(np.loadtxt(labels_fullpath, dtype=np.int64))

                    else:
                        # Load all the rest from the remeshed/sampled benchmark

                        mesh_fullpath = os.path.join(test_dirpath_base, method, "tr_reg_{}_{:03d}.ply".format(method, i))
                        labels_fullpath = os.path.join(test_dirpath_base, method, "tr_reg_{}_{:03d}.txt".format(method, i))
                        
                        verts, faces = pp3d.read_mesh(mesh_fullpath)
                        labels = torch.from_numpy(np.loadtxt(labels_fullpath, dtype=np.int64))
                        normals = None

                    # convert to torch
                    verts = torch.tensor(np.ascontiguousarray(verts)).float()
                    faces = torch.tensor(np.ascontiguousarray(faces))
                    
                    # center and scale
                    verts = diffusion_net.geometry.normalize_positions(verts, method='bbox')
                    
                    self.verts_list.append(verts)
                    self.faces_list.append(faces)
                    self.normals_list.append(normals)
                    self.labels_list.append(labels)
                    self.mut_list.append(method)

            

            print("loaded {} test meshes/clouds".format(len(self.verts_list)))
    

        # Precompute operators
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, normals=self.normals_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        # save to cache
        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.normals_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list, self.mut_list), load_cache)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.labels_list[idx], self.mut_list[idx]
