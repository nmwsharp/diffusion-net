import random
import numpy as np
import torch
import torch.nn as nn


class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        loss = torch.sum((a - b) ** 2, axis=(1, 2))
        return torch.mean(loss)


def normalize_area_scale(verts, faces):
    coords = verts[faces]
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]
    face_areas = torch.norm(torch.cross(vec_A, vec_B, dim=-1), dim=1) * 0.5
    total_area = torch.sum(face_areas)

    scale = (1 / torch.sqrt(total_area))
    verts = verts * scale

    # center
    verts = verts - verts.mean(dim=-2, keepdim=True)

    return verts


def euler_angles_to_rotation_matrix(theta, random_order=False):
    R_x = torch.tensor([[1, 0, 0], [0, torch.cos(theta[0]), -torch.sin(theta[0])], [0, torch.sin(theta[0]), torch.cos(theta[0])]])
    R_y = torch.tensor([[torch.cos(theta[1]), 0, torch.sin(theta[1])], [0, 1, 0], [-torch.sin(theta[1]), 0, torch.cos(theta[1])]])
    R_z = torch.tensor([[torch.cos(theta[2]), -torch.sin(theta[2]), 0], [torch.sin(theta[2]), torch.cos(theta[2]), 0], [0, 0, 1]])

    matrices = [R_x, R_y, R_z]
    if random_order:
        random.shuffle(matrices)
    R = torch.mm(matrices[2], torch.mm(matrices[1], matrices[0]))
    return R


def get_random_rotation(x, y, z):
    thetas = torch.zeros(3, dtype=torch.float)
    degree_angles = [x, y, z]
    for axis_ind, deg_angle in enumerate(degree_angles):
        rand_deg_angle = random.random() * 2 * deg_angle - deg_angle
        rand_radian_angle = float(rand_deg_angle * np.pi) / 180.0
        thetas[axis_ind] = rand_radian_angle

    return euler_angles_to_rotation_matrix(thetas, random_order=False)


def data_augmentation(verts, rot_x=0, rot_y=90, rot_z=0, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1):
    # # center
    # verts = verts - verts.mean(dim=-2, keepdim=True)

    # random rotation
    rotation_matrix = get_random_rotation(rot_x, rot_y, rot_z).to(verts.device)
    verts = verts @ rotation_matrix.T

    # random noise
    noise = std * torch.randn(verts.shape).to(verts.device)
    noise = noise.clamp(-noise_clip, noise_clip)
    verts += noise

    # random scaling
    scales = [scale_min, scale_max]
    scale = scales[0] + torch.rand((3,)) * (scales[1] - scales[0])
    verts = verts * scale.to(verts.device)

    return verts
