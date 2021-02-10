"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import pytest
from datasets import datasets
from autoencoder import AutoEncoder


class TestAutoencoder:
    @pytest.fixture(scope="module")
    def simple_shapes(self):
        batch_size = 4
        return datasets.SimpleShapes(batch_size, n_classes=10, n_rotations=3)

    def test_autoencoder(self, simple_shapes):
        n_epochs, learning_rate = 1, 0.001
        model = AutoEncoder(
            simple_shapes, device="cpu", n_epochs=n_epochs, learning_rate=learning_rate
        )
        model.run(stop_early=True)

    def test_autoencoder_with_shift_operator(self, simple_shapes):
        """Tests autoencoder with latent rotation"""
        n_epochs, learning_rate = 1, 0.001
        model = AutoEncoder(
            simple_shapes,
            device="cpu",
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            latent_operator_name="ShiftOperator",
        )
        model.run(stop_early=True)

    def test_autoencoder_with_disentangled_rotation(self, simple_shapes):
        """Tests autoencoder with latent rotation"""
        n_epochs, learning_rate = 1, 0.001
        model = AutoEncoder(
            simple_shapes,
            device="cpu",
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            latent_operator_name="DisentangledRotation",
        )
        model.run(stop_early=True)


class TestProjectiveMnistAutoencoder:
    def __init__(self):
        self.n_epochs = 1
        self.learning_rate = 0.01

    def test_standard_autoencoder(self, rotated_mnist):
        model = AutoEncoder(
            rotated_mnist, n_epochs=self.n_epochs, learning_rate=self.learning_rate
        )
        model.run(stop_early=True)

    def test_rotated_autoencoder(self, rotated_mnist):
        model = AutoEncoder(
            rotated_mnist,
            z_dim=400,
            latent_operator_name="DisentangledRotation",
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
        )
        model.run(stop_early=True)

    def test_shift_operator_autoencoder(self, rotated_mnist):
        model = AutoEncoder(
            rotated_mnist,
            z_dim=400,
            latent_operator_name="ShiftOperator",
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
        )
        model.run(stop_early=True)
