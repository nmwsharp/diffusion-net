import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import trimesh

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from human_segmentation_original_dataset import HumanSegOrigDataset


# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
args = parser.parse_args()


# system things
device = torch.device('cuda:0')
dtype = torch.float32

# problem/dataset things
n_class = 8

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
pretrain_path = os.path.join(base_path, "pretrained_models/human_seg_{}_4x128.pth".format(input_features))
model_save_path = os.path.join(base_path, "data/saved_models/human_seg_{}_4x128.pth".format(input_features))
dataset_path = os.path.join(base_path, "data/sig17_seg_benchmark")


# === Load datasets

# Load the test dataset
test_dataset = HumanSegOrigDataset(dataset_path, train=False, k_eig=k_eig, use_cache=False, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)

# Load the train dataset
if train:
    train_dataset = HumanSegOrigDataset(dataset_path, train=True, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)



# === Create the model

C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=n_class,
                                          C_width=128, 
                                          N_block=4, 
                                          last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                                          outputs_at='faces', 
                                          dropout=True)


model = model.to(device)

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))


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
            verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        # Apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

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


# borrow code from subdivnet
# https://github.com/lzhengning/SubdivNet/blob/master/subdivnet/utils.py
segment_colors = np.array([
    [0, 114, 189],
    [217, 83, 26],
    [238, 177, 32],
    [126, 47, 142],
    [117, 142, 48],
    [76, 190, 238],
    [162, 19, 48],
    [240, 166, 202],
])

def save_results(mesh_path, preds, labels):

    if not os.path.exists('results'):
        os.mkdir('results')

    labels = labels.cpu().numpy()
    preds  = preds.cpu().numpy()

    save_path = './results'

    mesh = trimesh.load_mesh(mesh_path, process=False)
    mesh_name = os.path.basename(mesh_path)[:-4]

    # print('mesh_path',mesh_path)
    # print('mesh_name',mesh_name)
    # print(save_path + '/pr-' + mesh_name + '.ply')

    mesh.visual.face_colors[:, :3] = segment_colors[preds]
    mesh.export(save_path + '/pr-' + mesh_name + '.ply')

    mesh.visual.face_colors[:, :3] = segment_colors[labels]
    mesh.export(save_path + '/gt-' + mesh_name + '.ply')

# Do an evaluation pass on the test dataset 
def test():
    
    model.eval()
    
    correct = 0
    total_num = 0
    with torch.no_grad():
    
        for data in tqdm(test_loader):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, mesh_file = data

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
            pred_labels = torch.max(preds, dim=1).indices

            # save
            print('mesh_file',mesh_file)
            print('pred',pred_labels.shape)
            print('labels',labels.shape)
            save_results(mesh_file,pred_labels,labels)

            # track accuracy
            
            this_correct = pred_labels.eq(labels).sum().item()
            this_num = labels.shape[0]
            correct += this_correct
            total_num += this_num

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
test_acc = test()
print("Overall test accuracy: {:06.3f}%".format(100*test_acc))
