import os
import os.path as osp
from os import listdir as osls
import shutil
import numpy as np
import progressbar

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.io import read_obj, read_ply

import geometry
from utils import toNP


## The following dataloader is adapted from: https://github.com/rubenwiersma/hsn

def edge_to_vertex_labels(faces, labels, n_nodes):
    """
    Converts a set of labels for edges to labels for vertices.
    :param faces: face indices of mesh
    :param labels: labels for edges
    :param n_nodes: number of nodes to map to
    """
    edge2key = set()
    edge_index = torch.LongTensor(0, 2)
    for face in faces.transpose(0, 1):
        edges = torch.stack([face[:2], face[1:], face[::2]], dim=0)
        for idx, edge in enumerate(edges):
            edge = edge.sort().values
            edges[idx] = edge
            if tuple(edge.tolist()) not in edge2key:
                edge2key.add(tuple(edge.tolist()))
                edge_index = torch.cat((edge_index, edge.view(1, -1)))

    res = torch.LongTensor(n_nodes).fill_(0)
    res[edge_index[:, 0]] = labels
    res[edge_index[:, 1]] = labels

    return res - 1

class ShapeSegGeometric(InMemoryDataset):
    r"""The Shape Segmentation dataset proposed by Maron et al. in
    "Convolutional neural networks on surfaces via seamless toric covers"
    <https://dl.acm.org/citation.cfm?id=3073616>`_ ,
    containing meshes from Adobe, SCAPE, FAUST, and MIT for training
    and SHREC shapes for testing.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Args:
        root (string): Root directory where the dataset should be saved.
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    mit_folders = {
        'crane': 18,
        'squat1': 25,
        'jumping': 15,
        'squat2': 25,
        'bouncing': 18,
        'march1': 25,
        'handstand': 18,
        'march2': 25
        }

    url = 'https://surfdrive.surf.nl/files/index.php/s/L68uSYpHtfO6dLa'

    def __init__(self, root, train=True, transform=None, pre_transform=None,
                 pre_filter=None):
        super(ShapeSegGeometric, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)


    @property
    def raw_file_names(self):
        # return ['ShapeSeg.zip'] # not configured right.
        return []

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download {} from {} and move it to {}'.
            format(self.raw_file_names, self.url, self.raw_dir))

    def process(self):
        # print('Extracting zip...')
        # extract_zip(self.raw_paths[0], self.raw_dir, log=False)
        shapeseg_path = osp.join(self.raw_dir, 'ShapeSeg')

        data_list = []
        #Adobe
        print('Processing Adobe')
        adobe_path = osp.join(shapeseg_path, 'Adobe', 'raw')
        extract_zip(osp.join(adobe_path, 'adobe.zip'), adobe_path)
        adobe_meshes = osp.join(adobe_path, 'meshes')
        adobe_meshes = osp.join(adobe_meshes, '{}.ply')
        adobe_segs = osp.join(adobe_path, 'segs', '{}.pt')
        for i in progressbar.progressbar(range(41)):
            data = read_ply(adobe_meshes.format(i))
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data.y = torch.load(adobe_segs.format(i))
            if hasattr(data, 'sample_idx'):
                data.y = data.y[data.sample_idx]
            data_list.append(data)
        torch.save(self.collate(data_list), osp.join(shapeseg_path, 'adobe.pt'))
            
        #FAUST
        print('Processing FAUST')
        faust_path = osp.join(shapeseg_path, 'FAUST', 'raw')
        extract_zip(osp.join(faust_path, 'faust.zip'), faust_path)
        faust_meshes = osp.join(faust_path, 'meshes')
        faust_meshes = osp.join(faust_meshes, 'tr_reg_{0:03d}.ply')
        faust_segs = torch.load(osp.join(faust_path, 'segs', 'faust_seg.pt'))
        for i in progressbar.progressbar(range(100)):
            data = read_ply(faust_meshes.format(i))
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data.y = faust_segs
            if hasattr(data, 'sample_idx'):
                data.y = data.y[data.sample_idx]
            data_list.append(data)
        torch.save(self.collate(data_list), osp.join(shapeseg_path, 'faust.pt'))

        #MIT
        print('Processing MIT')
        mit_path = osp.join(shapeseg_path, 'MIT', 'raw')
        extract_zip(osp.join(mit_path, 'mit.zip'), mit_path)
        mit_meshes = osp.join(mit_path, 'meshes')
        mit_seg = osp.join(mit_path, 'segs')
        for filename in progressbar.progressbar(osls(mit_meshes)):
            data = read_obj(osp.join(mit_meshes, filename))
            seg_path = osp.join(mit_seg, filename.replace('.obj', '.eseg'))
            segs = torch.from_numpy(np.loadtxt(seg_path)).long()
            data.y = edge_to_vertex_labels(data.face, segs, data.num_nodes)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        torch.save(self.collate(data_list), osp.join(shapeseg_path, 'mit.pt'))

        #SCAPE
        print('Processing SCAPE')
        scape_path = osp.join(shapeseg_path, 'SCAPE', 'raw')
        extract_zip(osp.join(scape_path, 'scape.zip'), scape_path)
        scape_meshes = osp.join(scape_path, 'meshes')
        scape_meshes = osp.join(scape_meshes, '{}.ply')
        scape_segs = torch.load(osp.join(scape_path, 'segs', 'scape_seg.pt'))
        for i in progressbar.progressbar(range(71)):
            data = read_ply(scape_meshes.format(i))
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data.y = scape_segs
            if hasattr(data, 'sample_idx'):
                data.y = data.y[data.sample_idx]
            data_list.append(data)
        torch.save(self.collate(data_list), osp.join(shapeseg_path, 'scape.pt'))

        torch.save(self.collate(data_list), self.processed_paths[0])
        data_list = []

        #SHREC
        print('Processing SHREC')
        shrec_path = osp.join(shapeseg_path, 'SHREC', 'raw')
        extract_zip(osp.join(shrec_path, 'shrec.zip'), shrec_path)
        shrec_meshes = osp.join(shrec_path, 'meshes')
        shrec_meshes = osp.join(shrec_meshes, '{}.ply')
        shrec_segs = osp.join(shrec_path, 'segs', '{}.pt')
        for i in progressbar.progressbar(range(18)):
            data = read_ply(shrec_meshes.format(i))
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data.y = torch.load(shrec_segs.format(i))
            if hasattr(data, 'sample_idx'):
                data.y = data.y[data.sample_idx]
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[1])


        shutil.rmtree(osp.join(self.raw_dir, 'ShapeSeg'))


class ShapeSegMeshDataset(torch.utils.data.Dataset):

    def __init__(self, opts, root_dir, train=True):
        self.opts = opts

        self.opts = opts
        self.train = train  # bool
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache")
        self.n_class = 8
        
        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []
        
        # Bootstrap off the geometric dataset (thanks for the parsing!)
        base_dataset = ShapeSegGeometric(root_dir, train)
        
        for data in base_dataset:

                verts = data.pos
                faces = data.face.transpose(0, 1).contiguous()
                labels = data.y

                # center and unit scale
                verts = geometry.normalize_positions(verts)

                self.verts_list.append(verts)
                self.faces_list.append(faces)
                self.labels_list.append(labels)
            
        # Precompute operators
        self.frames_list, self.massvec_list, self.evals_list, self.evecs_list, self.grad_from_spectral_list = geometry.get_all_operators(opts, self.verts_list, self.faces_list)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.evals_list[idx], self.evecs_list[idx], self.grad_from_spectral_list[idx], self.labels_list[idx]

