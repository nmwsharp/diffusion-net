import os
import sys
from itertools import permutations, combinations
import numpy as np

import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import toNP

class FaustScapeDataset(Dataset):
    def __init__(self, root_dir, name="faust", train=True, k_eig=128, n_fmap=30, use_cache=True, op_cache_dir=None):

        # NOTE: These datasets are setup such that each dataset object always loads the entire dataset regardless of train/test mode. The correspondence pair combinations are then set such that the train dataset only returns train pairs, and the test dataset only returns test pairs. Be aware of this if you try to adapt the code for any other purpose!

        self.train = train  # bool
        self.k_eig = k_eig
        self.n_fmap = n_fmap
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, name, "cache")
        self.op_cache_dir = op_cache_dir
        self.name = name

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.vts_list = []
        self.names_list = []

        # set combinations
        n_train = {'faust':80, 'scape':51}[self.name]
        if self.train:
            self.combinations = list(permutations(range(n_train), 2))
        else:
            self.combinations = list(combinations(range(n_train, n_train + 20), 2))

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            load_cache = train_cache if self.train else test_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.hks_list,
                    self.vts_list,
                    self.names_list
                ) = torch.load(load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        mesh_files = []
        vts_files = []

        # load faust data
        mesh_dirpath = os.path.join(self.root_dir, name, "off_2")
        vts_dirpath = os.path.join(self.root_dir, name, "corres")
        for fname in os.listdir(mesh_dirpath):
            mesh_fullpath = os.path.join(mesh_dirpath, fname)
            vts_fullpath = os.path.join(vts_dirpath, fname[:-4] + ".vts")
            mesh_files.append(mesh_fullpath)
            vts_files.append(vts_fullpath)

        print("loading {} meshes".format(len(mesh_files)))

        mesh_files, vts_files = sorted(mesh_files), sorted(vts_files)

        # Load the actual files
        for iFile in range(len(mesh_files)):

            print("loading mesh " + str(mesh_files[iFile]))

            verts, faces = pp3d.read_mesh(mesh_files[iFile])
            vts_file = np.loadtxt(vts_files[iFile]).astype(int) - 1 # convert from 1-based indexing

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            vts_file = torch.tensor(np.ascontiguousarray(vts_file))

            # center and unit-area scale
            verts = diffusion_net.geometry.normalize_positions(verts, faces=faces, scale_method='area')

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.vts_list.append(vts_file)
            self.names_list.append(os.path.basename(mesh_files[iFile]).split(".")[0])

        for ind, labels in enumerate(self.vts_list):
            self.vts_list[ind] = labels

        # Precompute operators
        (
            self.frames_list,
            self.massvec_list,
            self.L_list,
            self.evals_list,
            self.evecs_list,
            self.gradX_list,
            self.gradY_list,
        ) = diffusion_net.geometry.get_all_operators(
            self.verts_list,
            self.faces_list,
            k_eig=self.k_eig,
            op_cache_dir=self.op_cache_dir,
        )

        self.hks_list = [diffusion_net.geometry.compute_hks_autoscale(self.evals_list[i], self.evecs_list[i], 16)
                         for i in range(len(self.L_list))]

        # save to cache
        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save(
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.hks_list,
                    self.vts_list,
                    self.names_list,
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx1, idx2 = self.combinations[idx]

        shape1 = [
            self.verts_list[idx1],
            self.faces_list[idx1],
            self.frames_list[idx1],
            self.massvec_list[idx1],
            self.L_list[idx1],
            self.evals_list[idx1],
            self.evecs_list[idx1],
            self.gradX_list[idx1],
            self.gradY_list[idx1],
            self.hks_list[idx1],
            self.vts_list[idx1],
            self.names_list[idx1],
        ]

        shape2 = [
            self.verts_list[idx2],
            self.faces_list[idx2],
            self.frames_list[idx2],
            self.massvec_list[idx2],
            self.L_list[idx2],
            self.evals_list[idx2],
            self.evecs_list[idx2],
            self.gradX_list[idx2],
            self.gradY_list[idx2],
            self.hks_list[idx2],
            self.vts_list[idx2],
            self.names_list[idx2],
        ]

        # Compute the ground-truth functional map between the pair
        vts1, vts2 = shape1[10], shape2[10]
        evec_1, evec_2 = shape1[6][:, :self.n_fmap], shape2[6][:, :self.n_fmap]
        evec_1_a, evec_2_a = evec_1[vts1,:], evec_2[vts2,:]
        solve_out = torch.lstsq(evec_2_a, evec_1_a)[0] # TODO replace with torch.linalg version in future torch
        C_gt = solve_out[:evec_1_a.size(-1)].t()
        resids = solve_out[evec_1_a.size(-1):]

        # Alternately, do it with numpy instead:
        # solve_out = np.linalg.lstsq(toNP(evec_1_a), toNP(evec_2_a), rcond=None)
        # C_gt = solve_out[0]
        # C_gt = torch.Tensor(C_gt.T)
        # resids = torch.tensor(solve_out[1])
        
        return (shape1, shape2, C_gt)
