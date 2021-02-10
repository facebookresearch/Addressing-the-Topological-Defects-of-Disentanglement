"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import numpy as np
import functools
import pdb


class ShiftOperator:
    """Performs discrete shift based on n_rotations."""

    def __init__(self, n_rotations, device):
        self.n_rotations = n_rotations
        self.device = device
        self.translation_matrices = self.generate_shift_operator_matrices(
            n_rotations + 1
        )

    def __call__(self, z_batch, angles):
        """Interface for Autoencoder"""
        return self.translate_batch(z_batch, angles)

    def translate_batch(self, z_batch, angles):
        """Applies shift operator to batch

        Args:
            angles (array of floats): counter-clockwise rotation in degrees.
        """
        smallest_angle = 360 / (self.n_rotations + 1)

        if angles.dim() > 1:
            shifts = angles[:, 0] / smallest_angle
        else:
            shifts = angles / smallest_angle
        try:
            translated_batch = [
                self.translate(z, shifts[i].long()) for i, z in enumerate(z_batch)
            ]
        except IndexError as e:
            print("===ANGLES ARE", angles)
            raise e
        return torch.stack(translated_batch)

    def translate(self, z, shift):
        """Translate latent

        Args:
            z (1-dim tensor): latent vector
            shift (int): amount by which to shift.
                shift of 0 corresponds to the identity.
        """
        # reshape into 2D tensor
        z_2d = z.reshape(self.n_rotations + 1, -1)
        translation_matrix = self.translation_matrices[shift]
        # move to cpu if tensor is cpu. Used for eval
        if not z_2d.is_cuda:
            translation_matrix = translation_matrix.cpu()
        # translation
        z_2d_shifted = translation_matrix.matmul(z_2d)
        # reshape back
        z_shifted = z_2d_shifted.reshape(z.shape)
        return z_shifted

    def generate_shift_operator_matrices(self, n_rotations):
        """Generates family of shift operator matrices"""
        translation_matrix = np.zeros((n_rotations, n_rotations))
        for i in range(n_rotations):
            translation_matrix[i, (i + 1) % n_rotations] = 1

        translation_matrices = [np.eye(n_rotations, n_rotations)]

        T = np.eye(n_rotations, n_rotations)

        for i in range(n_rotations - 1):
            T = np.dot(translation_matrix, T)
            translation_matrices.append(T)

        translation_matrices = np.array(translation_matrices)
        _translation_matrices = torch.tensor(
            translation_matrices, dtype=torch.float32, device=self.device,
        )
        return _translation_matrices


class ComplexShiftOperator:
    """Performs discrete shift based on n_rotations"""

    def __init__(self, cardinals, z_dim, device, unique_transfo=False, index=None):
        self.cardinals = cardinals
        self.z_dim = z_dim
        self.device = device
        self.translation_matrices = self.generate_translation_matrices(
            self.cardinals, self.z_dim
        )
        if unique_transfo:
            if (np.array(cardinals)>1).sum()==1:
                self.index = int((np.array(cardinals)>1).nonzero()[0])
            elif (np.array(cardinals)>1).sum()>1:
                if index is None:
                    print("Must provide the index of the operator !")
                else:
                    self.index = index
            self.translate_batch = self.translate_batch_unique
        else:
            self.translate_batch = self.translate_batch_multiple

    def __call__(self, z_batch, shifts):
        """Interface for Autoencoder"""
        z_batch_r, z_batch_i = z_batch
        return self.translate_batch(z_batch_r, z_batch_i, shifts)

    def translate_batch_unique(self, z_batch_r, z_batch_i, shifts):
        """Translates batch in the case of a unique transformations (Faster)"""
        tr = self.translation_matrices[self.index][0][shifts[:, 0]]
        ti = self.translation_matrices[self.index][1][shifts[:, 0]]
        z_batch_r_shifted = tr * z_batch_r - ti * z_batch_i
        z_batch_i_shifted = tr * z_batch_i + ti * z_batch_r
        return (
            z_batch_r_shifted,
            z_batch_i_shifted,
        )  

    def translate_batch_multiple(self, z_batch_r, z_batch_i, shifts):
        """Translates batch in the case of multiple transformations"""

        (Mr, Mi) = self.build_multipliers(shifts)
        z_batch_r_shifted = Mr * z_batch_r - Mi * z_batch_i
        z_batch_i_shifted = Mr * z_batch_i + Mi * z_batch_r
        return (
            z_batch_r_shifted,
            z_batch_i_shifted,
        )  

    def build_multipliers(self, shifts):
        size_batch, n_transfo = shifts.shape
        Mr = torch.ones((size_batch, self.z_dim), device=self.device)
        Mi = torch.zeros((size_batch, self.z_dim), device=self.device)
        for i in range(n_transfo):
            tr = self.translation_matrices[i][0][shifts[:, i]]
            ti = self.translation_matrices[i][1][shifts[:, i]]
            Mr = Mr * tr - Mi * ti
            Mi = Mr * ti + Mi * tr
        return (Mr, Mi)

    def translate(self, zr, zi, shift):
        """Translate latent

        Args:
            z (1-dim tensor): latent vector
            shift (int): amount by which to shift
        """
        for i in range(len(shift)):
            tr = self.translation_matrices[i][0][shift[i]]
            ti = self.translation_matrices[i][1][shift[i]]
            zr = zr * tr - zi * ti
            zi = zi * tr + zr * ti
        return (zr, zi)

    def generate_translation_matrices(self, cardinals, z_dim):
        """Generates family of translation matrices"""

        def DFT_matrix(cardinal, z_dim):
            i, j = np.meshgrid(np.arange(cardinal), np.arange(cardinal))
            omega = np.exp(2 * np.pi * 1j / cardinal)
            W = np.power(omega, i * j)
            return W

        # Loop over all transformations that can happen to the sample
        XYZ = []
        for i, t in enumerate(cardinals):
            K = self.cardinals[i]
            X_i =  np.arange(K)
            if z_dim % K: # creates in shift operator an unfinished cycle
                second_dim = (
                    int(np.floor(z_dim / K)) + 1
                )  # TODO: not sure this is the right way
            else: # creates in shift operator a finished cycle
                second_dim = int(z_dim / K)
            
            X_i = np.tile(X_i.flatten(), (second_dim))[:z_dim]
            XYZ.append(X_i)

        _all_translation_matrices = list()
        for i in range(len(cardinals)):
            translation_matrices = DFT_matrix(cardinals[i], z_dim)
            translation_matrices = translation_matrices[:, XYZ[i]]
            translation_matrices_r = np.real(translation_matrices)
            translation_matrices_i = np.imag(translation_matrices)
            _translation_matrices_r = torch.tensor(
                translation_matrices_r, dtype=torch.float32, device=self.device,
            )
            _translation_matrices_i = torch.tensor(
                translation_matrices_i, dtype=torch.float32, device=self.device,
            )

            _all_translation_matrices.append(
                (_translation_matrices_r, _translation_matrices_i,)
            )
        return _all_translation_matrices


class DisentangledRotation:
    """Performs rotation using rotation matrix of the form:
        [cos, -sin], [sin, cos]

    Args:
        n_rotations (int): discrete rotations needed before identity is reached
    """

    def __init__(self, n_rotations, device):
        self.n_rotations = n_rotations
        self.device = device

    def __call__(self, z, angles):
        """Interface for Autoencoder"""
        return self.rotate_batch(z, angles)

    def rotate_batch(self, x_batch, angles):
        """Rotates batch"""
        rotated_batch = []
        if angles.dim() > 1:
            angles = angles[:, 0]
        else:
            angles = angles
        for i, x in enumerate(x_batch):
            x_rotated = self.rotate(x, angles[i])
            rotated_batch.append(x_rotated)
        return torch.stack(rotated_batch)

    def rotate(self, x, angle):
        """Clockwise rotation or translation
        Args:
            x (1D tensor): representing latent vector
            angle (float): rotation angle in degrees
        Returns: rotated tensor of same shape as x
        """
        if x.dim() != 1:
            raise ValueError(f"x must be a flattened 1D vector. Got shape {x.shape}")
        rotation_matrix = self.get_rotation_matrix(angle, x.shape[0])
        if not x.is_cuda:
            rotation_matrix = rotation_matrix.cpu()
        x_rotated = rotation_matrix.matmul(x)
        return x_rotated

    @functools.lru_cache()
    def get_rotation_matrix(self, angle, dim):
        """Angle is the rotation angle in degrees.
        Returns rotation matrix that operates on first two dimensions
        """
        rotation_matrix = torch.diag(torch.ones(dim, device=self.device))
        if angle == 0.0:
            return rotation_matrix
        radians = (angle / 360) * torch.tensor(2 * np.pi)
        matrix_2d = torch.tensor(
            [
                [torch.cos(radians), -torch.sin(radians)],
                [torch.sin(radians), torch.cos(radians)],
            ]
        )
        rotation_matrix[0][0] = matrix_2d[0][0]
        rotation_matrix[0][1] = matrix_2d[0][1]
        rotation_matrix[1][0] = matrix_2d[1][0]
        rotation_matrix[1][1] = matrix_2d[1][1]
        return rotation_matrix.to(device=self.device)
