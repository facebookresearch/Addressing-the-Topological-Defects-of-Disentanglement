"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import pytest
from datasets import datasets
from cci_variational_autoencoder import CCIVariationalAutoEncoder

BATCH_SIZE = 16


@pytest.fixture(scope="module")
def rotated_mnist():
    rotated_mnist = datasets.ProjectiveMNIST(
        BATCH_SIZE,
        n_rotations=9,
        train_set_proportion=0.001,
        test_set_proportion=0.001,
        valid_set_proportion=0.001,
    )
    return rotated_mnist


@pytest.fixture(scope="module")
def simple_shapes():
    batch_size = 16
    return datasets.SimpleShapes(batch_size, n_classes=10)


class TestCCIVariationalAutoEncoder:
    def test_vae(self, simple_shapes):
        n_epochs, learning_rate = 1, 0.001
        model = CCIVariationalAutoEncoder(
            simple_shapes,
            beta=1.0,
            c_max=0.0,
            device="cpu",
            n_epochs=n_epochs,
            learning_rate=learning_rate,
        )
        model.train()

    def test_beta_vae(self, simple_shapes):
        n_epochs, learning_rate = 1, 0.001
        model = CCIVariationalAutoEncoder(
            simple_shapes,
            beta=1.0,
            c_max=0.0,
            device="cpu",
            n_epochs=n_epochs,
            learning_rate=learning_rate,
        )
        model.train()

    def test_cci_vae(self, simple_shapes):
        n_epochs, learning_rate = 1, 0.001
        model = CCIVariationalAutoEncoder(
            simple_shapes,
            beta=100.0,
            c_max=36.0,
            device="cpu",
            n_epochs=n_epochs,
            learning_rate=learning_rate,
        )
        model.train()


class TestProjectiveMNISTVAE:
    def test_vae(self, rotated_mnist):
        n_epochs, learning_rate = 1, 0.001
        model = CCIVariationalAutoEncoder(
            rotated_mnist,
            beta=1.0,
            c_max=0.0,
            device="cpu",
            n_epochs=n_epochs,
            learning_rate=learning_rate,
        )
        model.train(stop_early=True)

    def test_cci_vae(self, rotated_mnist):
        n_epochs, learning_rate = 1, 0.001
        model = CCIVariationalAutoEncoder(
            rotated_mnist,
            beta=100.0,
            c_max=36.0,
            device="cpu",
            n_epochs=n_epochs,
            learning_rate=learning_rate,
        )
        model.train(stop_early=True)
