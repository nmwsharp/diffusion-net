import os
import sys
import unittest

import scipy
import scipy.sparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/"))  # add the path to the DiffusionNet src
from diffusion_net.geometry import build_grad, build_tangent_frames, edge_tangent_vectors, find_knn
from diffusion_net.utils import toNP

import potpourri3d as pp3d


def build_grad_reference(verts, edges, edge_tangent_vectors):
    """
    The original per-vertex loop implementation of build_grad(), kept here as a reference.
    """

    edges_np = toNP(edges)
    edge_tangent_vectors_np = toNP(edge_tangent_vectors)

    # Build outgoing neighbor lists
    N = verts.shape[0]
    vert_edge_outgoing = [[] for i in range(N)]
    for iE in range(edges_np.shape[1]):
        tail_ind = edges_np[0, iE]
        tip_ind = edges_np[1, iE]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(iE)

    # Build local inversion matrix for each vertex
    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5
    for iV in range(N):
        n_neigh = len(vert_edge_outgoing[iV])

        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [iV]
        for i_neigh in range(n_neigh):
            iE = vert_edge_outgoing[iV][i_neigh]
            jV = edges_np[1, iE]
            ind_lookup.append(jV)

            edge_vec = edge_tangent_vectors[iE][:]
            w_e = 1.

            lhs_mat[i_neigh][:] = w_e * edge_vec
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

        lhs_T = lhs_mat.T
        lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(2)) @ lhs_T

        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(iV)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)), shape=(
            N, N)).tocsc()

    return mat


def icosphere(n_subdiv):
    # numpy out: (V,3) verts, (F,3) faces of a subdivided icosahedron on the unit sphere
    t = (1. + np.sqrt(5.)) / 2.
    verts = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
             [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
             [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
             [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
             [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
             [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    verts = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in verts]

    for _ in range(n_subdiv):
        midpoint_cache = {}

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in midpoint_cache:
                m = verts[i] + verts[j]
                verts.append(m / np.linalg.norm(m))
                midpoint_cache[key] = len(verts) - 1
            return midpoint_cache[key]

        new_faces = []
        for (a, b, c) in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces

    return np.array(verts), np.array(faces, dtype=np.int64)


def grid(n, m):
    # numpy out: (V,3) verts, (F,3) faces of a slightly curved, randomly perturbed planar grid (has boundary)
    rng = np.random.RandomState(0)
    xs, ys = np.meshgrid(np.linspace(0., 1., n), np.linspace(0., 1., m), indexing='ij')
    xs = xs + 0.2 / n * (rng.rand(n, m) - 0.5)
    ys = ys + 0.2 / m * (rng.rand(n, m) - 0.5)
    verts = np.stack((xs, ys, 0.3 * np.sin(3. * xs) * np.cos(2. * ys)), axis=-1).reshape(-1, 3)
    inds = np.arange(n * m).reshape(n, m)
    v00, v10, v01, v11 = inds[:-1, :-1], inds[1:, :-1], inds[:-1, 1:], inds[1:, 1:]
    faces = np.concatenate((np.stack((v00, v10, v11), axis=-1).reshape(-1, 3),
                            np.stack((v00, v11, v01), axis=-1).reshape(-1, 3)), axis=0)
    return verts, faces


def mesh_grad_inputs(verts_np, faces_np):
    # Same edges and tangent vectors as compute_operators() uses for meshes
    verts = torch.tensor(verts_np, dtype=torch.float64)
    faces = torch.tensor(faces_np)
    frames = build_tangent_frames(verts, faces)
    L = pp3d.cotan_laplacian(verts_np, faces_np, denom_eps=1e-10)
    L_coo = L.tocoo()
    edges = torch.tensor(np.stack((L_coo.row, L_coo.col), axis=0), dtype=faces.dtype)
    edge_vecs = edge_tangent_vectors(verts, frames, edges)
    return verts, edges, edge_vecs


class TestBuildGrad(unittest.TestCase):

    def assert_grad_equal(self, verts, edges, edge_vecs, rtol=1e-10):
        mat = build_grad(verts, edges, edge_vecs)
        mat_ref = build_grad_reference(verts, edges, edge_vecs)

        self.assertEqual(mat.shape, mat_ref.shape)
        self.assertEqual(mat.dtype, mat_ref.dtype)

        # same sparsity pattern
        np.testing.assert_array_equal(mat.indptr, mat_ref.indptr)
        np.testing.assert_array_equal(mat.indices, mat_ref.indices)

        # same values, up to floating point roundoff
        scale = np.max(np.abs(mat_ref.data))
        max_err = np.max(np.abs(mat.data - mat_ref.data)) if mat.nnz > 0 else 0.
        self.assertLessEqual(max_err, rtol * scale)

        # and the same action on a smooth function
        verts_np = verts if isinstance(verts, np.ndarray) else toNP(verts)
        f = verts_np[:, 0] ** 2 + np.sin(verts_np[:, 1])
        np.testing.assert_allclose(mat @ f, mat_ref @ f, rtol=0., atol=rtol * scale * np.max(np.abs(f)) * 10)

    def test_icosphere(self):
        verts_np, faces_np = icosphere(3)
        self.assert_grad_equal(*mesh_grad_inputs(verts_np, faces_np))

    def test_grid_with_boundary(self):
        verts_np, faces_np = grid(25, 20)
        self.assert_grad_equal(*mesh_grad_inputs(verts_np, faces_np))

    def test_point_cloud(self):
        # Same edges and tangent vectors as build_grad_point_cloud() uses
        verts_np, _ = icosphere(2)
        verts = torch.tensor(verts_np, dtype=torch.float64)
        frames = build_tangent_frames(verts, torch.zeros((0, 3), dtype=torch.int64))
        n_neighbors_cloud = 30
        _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
        edge_inds_from = np.repeat(np.arange(verts.shape[0]), n_neighbors_cloud)
        edges = np.stack((edge_inds_from, toNP(neigh_inds).flatten()))
        edge_vecs = edge_tangent_vectors(verts, frames, edges)
        self.assert_grad_equal(verts_np, torch.tensor(edges), edge_vecs)

    def test_degenerate_edges(self):
        # Self-loops, duplicate edges, an isolated vertex, and single precision tangent vectors
        rng = np.random.RandomState(0)
        N = 50
        tails = rng.randint(0, N - 1, size=400)
        tips = rng.randint(0, N - 1, size=400)
        tips[:20] = tails[:20]  # self-loops
        edges = np.concatenate((np.stack((tails, tips)), np.stack((tails[20:40], tips[20:40]))), axis=1)  # duplicates
        verts = torch.tensor(rng.rand(N, 3))  # the last vertex has no edges
        edge_vecs = torch.tensor(rng.randn(edges.shape[1], 2), dtype=torch.float32)
        self.assert_grad_equal(verts, torch.tensor(edges), edge_vecs)


if __name__ == "__main__":
    unittest.main()
