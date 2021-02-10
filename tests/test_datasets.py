"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import math
from datasets import transformations
from datasets import datasets


class TestSimpleShapes:
    def test_train_loader(self):
        simple_shapes = datasets.SimpleShapes(16, n_classes=3)
        assert hasattr(simple_shapes, "train_loader")
        assert hasattr(simple_shapes, "test_loader")
        assert len(simple_shapes.train_loader) > 0
        assert len(simple_shapes.test_loader) > 0

    def test_transformations(self):
        simple_shapes = datasets.SimpleShapes(
            16,
            n_classes=3,
            n_rotations=9,
            n_x_translations=5,
            n_y_translations=10,
            scaling_factors=(1.0, 1.2),
        )
        assert simple_shapes.total_n_transformations > 50


class TestProjectiveMNIST:
    def test_creation(self):
        """Verifies rotated mnist is created properly"""
        n_rotations = 9
        batch_size = 16
        train_size = 5000

        rotated_mnist = datasets.ProjectiveMNIST(batch_size, n_rotations=n_rotations)
        expected_n_batches = math.ceil(
            (rotated_mnist.total_n_transformations ** 2) * train_size / batch_size
        )
        assert len(rotated_mnist.train_loader) == expected_n_batches

        # test shape of x2
        assert rotated_mnist.X_train[3][1].shape == torch.Size([1, 28, 28])

    def test_proportion(self):
        n_rotations = 9
        batch_size = 16
        train_proportion = 0.001
        test_proportion = 0.005
        # 10k for validation
        full_train_size = 50000
        full_test_size = 10000

        rotated_mnist = datasets.ProjectiveMNIST(
            batch_size,
            n_rotations=n_rotations,
            train_set_proportion=train_proportion,
            valid_set_proportion=train_proportion,
            test_set_proportion=test_proportion,
        )

        expected_train_size = (
            full_train_size * train_proportion * (n_rotations + 1) ** 2
        )
        expected_test_size = full_test_size * test_proportion * (n_rotations + 1) ** 2

        assert len(rotated_mnist.X_train) == expected_train_size
        assert len(rotated_mnist.X_test) == expected_test_size


class TestTransformations:
    def test_transform(self):
        shape = (1, 30, 30)
        image = torch.rand(shape)
        params = transformations.Params(angle=45.0)

        rotated_X = transformations.transform(image, params)
        assert torch.is_tensor(rotated_X)
        assert rotated_X.shape == image.shape
