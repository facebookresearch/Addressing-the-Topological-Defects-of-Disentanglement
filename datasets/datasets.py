"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision
from . import data_utils
from abc import ABC, abstractmethod
from datasets import transformations
import numpy as np
import random
import json


class AbstractDataset(ABC):
    """
    Defines common fields needed for datasets

    Attributes:
        batch_size (int): batch size used for dataloaders
        train_load (torch.utils.data.Dataset): X1, X2, Angle(s)
        test_load (torch.utils.data.Dataset): X1, X2, Angle(s)
        pairs (bool): indicates whether to use Pairs dataset where both x1 and x2 are transformed.
            Otherwise, Single dataset is used where only x1 is transformed.
    """

    def __init__(
        self,
        batch_size,
        n_rotations=0,
        n_x_translations=0,
        n_y_translations=0,
        scaling_factors=(1.0,),
        seed=0,
        pairs=True,
    ):
        AbstractDataset.set_seed(seed)
        self.batch_size = batch_size
        self.n_x_translations, self.n_y_translations = (
            n_x_translations,
            n_y_translations,
        )
        self.n_rotations, self.scaling_factors = n_rotations, scaling_factors
        self.X_orig_train, self.X_orig_valid, self.X_orig_test = self.get_original()

        self.transform_params = list(
            transformations.get_transform_params(
                n_rotations=self.n_rotations,
                n_x_translations=self.n_x_translations,
                n_y_translations=self.n_y_translations,
                scaling_factors=self.scaling_factors,
            )
        )

        data_cls = Pairs if pairs else Single

        self.X_train = data_cls(self.X_orig_train, self.transform_params)
        self.train_loader = torch.utils.data.DataLoader(
            self.X_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=Pairs.collate,
        )
        #  For validation and test, use shuffle = False to have SequentialSampler(dataset) by default
        # (see https://github.com/pytorch/pytorch/blob/bfa94487b968ccb570ef8cd9547029b967e76ed0/torch/utils/data/dataloader.py#L257)
        self.X_valid = data_cls(self.X_orig_valid, self.transform_params)
        self.valid_loader = torch.utils.data.DataLoader(
            self.X_valid,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=Pairs.collate,
        )

        self.X_test = data_cls(self.X_orig_test, self.transform_params)
        self.test_loader = torch.utils.data.DataLoader(
            self.X_test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=Pairs.collate,
        )

        self.test_loader_batch_1 = torch.utils.data.DataLoader(
            self.X_test, batch_size=1, shuffle=False, collate_fn=Pairs.collate,
        )
        self.test_loader_batch_100 = torch.utils.data.DataLoader(
            self.X_test, batch_size=100, shuffle=False, collate_fn=Pairs.collate,
        )

    def __repr__(self):
        attributes = {
            "n_rotations": self.n_rotations,
            "n_x_translations": self.n_x_translations,
            "n_y_translations": self.n_y_translations,
            "scaling_factors": self.scaling_factors,
        }
        return json.dumps(attributes)

    @abstractmethod
    def get_original(self):
        """Sets X_train and X_test to images in original dataset"""
        pass

    @property
    def total_n_transformations(self):
        """Computes the total number of transformations"""
        n_translations = (1 + self.n_x_translations) * (1 + self.n_y_translations)
        n = n_translations * (1 + self.n_rotations) * len(self.scaling_factors)
        return n

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @classmethod
    def __subclasshook__(cls, C):
        """Verifies dataset has loader of correct type"""
        for loader in ["train_loader", "test_loader"]:
            is_valid = hasattr(cls, loader) and isinstance(
                (getattr(cls, loader)), Dataset
            )
            if not is_valid:
                return False
        return True


class ProjectiveMNIST(AbstractDataset):
    """Builds MNIST dataset with transformations applied lazly.
    Loader contains: (digit, rotated_digit, angle)
    Shape of Data: (batch_size, 1, 28, 28)
    Args:
        batch_size (int): batch size to user for dataloaders
        n_rotations (int): number discrete rotations per image
        train_set_proportion (float): proportion of training set to keep
        valid_set_proportion (float): proportion of training set to keep
        test_set_proportion (float): proportion of training set to keep
    """

    def __init__(
        self,
        batch_size,
        n_rotations=4,
        n_x_translations=0,
        n_y_translations=0,
        scaling_factors=(1.0,),
        train_set_proportion=0.1,
        valid_set_proportion=1.0,
        test_set_proportion=1.0,
        seed=0,
        pairs=True,
    ):
        self.train_set_proportion = train_set_proportion
        self.valid_set_proportion = valid_set_proportion
        self.test_set_proportion = test_set_proportion

        super().__init__(
            batch_size,
            n_rotations,
            n_x_translations,
            n_y_translations,
            scaling_factors,
            seed,
            pairs,
        )
        self.n_pixels = self.X_orig_train[0].shape[1]
        self.n_channels = 1

    def get_original(self):
        """Returns original training and test images"""
        mnist_train, mnist_val, mnist_test = self.download_mnist()
        # normalize MNIST so values are between [0, 1]
        x_train = mnist_train.data.unsqueeze(1) / 255.0
        x_val = mnist_val.data.unsqueeze(1) / 255.0
        x_test = mnist_test.data.unsqueeze(1) / 255.0
        return x_train, x_val, x_test

    @staticmethod
    def stratified_sample(X, y, size):
        """Returns a stratified sample"""
        if size == 1.0:
            return X
        test_size = 1 - size
        sampler = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=0
        )
        indices, _ = next(sampler.split(X, y))
        X_sample = X[indices]
        return X_sample

    @staticmethod
    def split_train_valid(train_set, split=10000):
        num_train = len(train_set)
        indices = list(range(num_train))

        train_idx, valid_idx = indices[split:], indices[:split]
        train_data = train_set.data[train_idx]
        valid_data = train_set.data[valid_idx]

        train_targets = train_set.targets[train_idx]
        valid_targets = train_set.targets[valid_idx]

        return train_data, train_targets, valid_data, valid_targets

    def download_mnist(self):
        """Skips download if cache is available"""
        train_set = torchvision.datasets.MNIST(
            "/tmp/",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        test_set = torchvision.datasets.MNIST(
            "/tmp/",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        (
            train_data,
            train_targets,
            valid_data,
            valid_targets,
        ) = ProjectiveMNIST.split_train_valid(train_set)
        # stratified samples
        train_data = ProjectiveMNIST.stratified_sample(
            train_data, train_targets, self.train_set_proportion
        )
        valid_data = ProjectiveMNIST.stratified_sample(
            valid_data, valid_targets, self.valid_set_proportion
        )
        test_data = ProjectiveMNIST.stratified_sample(
            test_set.data, test_set.targets, self.test_set_proportion
        )

        return train_data, valid_data, test_data


class ProjectiveSingleDigitMNIST(AbstractDataset):
    """Builds MNIST dataset with transformations applied lazly.
    Loader contains: (digit, rotated_digit, angle)
    Shape of Data: (batch_size, 1, 28, 28)
    Args:
        batch_size (int): batch size to user for dataloaders
        n_rotations (int): number discrete rotations per image
        train_set_proportion (float): proportion of training set to keep
        valid_set_proportion (float): proportion of training set to keep
        test_set_proportion (float): proportion of training set to keep
    """

    def __init__(
        self,
        batch_size,
        n_rotations=4,
        n_x_translations=0,
        n_y_translations=0,
        scaling_factors=(1.0,),
        train_set_proportion=0.1,
        valid_set_proportion=1.0,
        test_set_proportion=1.0,
        seed=0,
        pairs=True,
        digit=4,
    ):
        self.train_set_proportion = train_set_proportion
        self.valid_set_proportion = valid_set_proportion
        self.test_set_proportion = test_set_proportion
        self.digit = digit

        super().__init__(
            batch_size,
            n_rotations,
            n_x_translations,
            n_y_translations,
            scaling_factors,
            seed,
            pairs,
        )
        self.n_pixels = self.X_orig_train[0].shape[1]
        self.n_channels = 1

    def get_original(self):
        """Returns original training and test images"""
        mnist_train, mnist_val, mnist_test = self.download_mnist()
        # normalize MNIST so values are between [0, 1]
        x_train = mnist_train.data.unsqueeze(1) / 255.0
        x_val = mnist_val.data.unsqueeze(1) / 255.0
        x_test = mnist_test.data.unsqueeze(1) / 255.0
        return x_train, x_val, x_test

    @staticmethod
    def split_train_valid(train_set, split=10000):
        num_train = len(train_set)
        indices = list(range(num_train))

        train_idx, valid_idx = indices[split:], indices[:split]
        train_data = train_set.data[train_idx]
        valid_data = train_set.data[valid_idx]

        train_targets = train_set.targets[train_idx]
        valid_targets = train_set.targets[valid_idx]

        return train_data, train_targets, valid_data, valid_targets

    def sample_single_digit(self, x, targets, proportion):
        idx = targets == self.digit
        x_digit = x[idx]
        sample_size = int(len(idx) * proportion)
        return x_digit[:sample_size]

    def download_mnist(self):
        """Skips download if cache is available"""
        train_set = torchvision.datasets.MNIST(
            "/tmp/",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        test_set = torchvision.datasets.MNIST(
            "/tmp/",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        (
            train_data,
            train_targets,
            valid_data,
            valid_targets,
        ) = ProjectiveMNIST.split_train_valid(train_set)
        # stratified samples
        train_data = self.sample_single_digit(
            train_data, train_targets, self.train_set_proportion
        )
        valid_data = self.sample_single_digit(
            valid_data, valid_targets, self.valid_set_proportion
        )
        test_data = self.sample_single_digit(
            test_set.data, test_set.targets, self.test_set_proportion
        )
        return train_data, valid_data, test_data


class SimpleShapes(AbstractDataset):
    def __init__(
        self,
        batch_size,
        n_pixels=28,
        n_classes=300,
        n_points=5,
        n_rotations=9,
        n_x_translations=0,
        n_y_translations=0,
        scaling_factors=(1.0,),
        n_channels=1,
        seed=0,
        pairs=True,
    ):
        self.n_pixels, self.n_classes = n_pixels, n_classes
        self.n_points, self.n_channels = n_points, n_channels

        super().__init__(
            batch_size,
            n_rotations,
            n_x_translations,
            n_y_translations,
            scaling_factors,
            seed,
            pairs,
        )

    @staticmethod
    def normalize(X):
        return torch.clamp(X + 1, 0.0, 1.0)

    def get_original(self):
        np.random.seed(1)  # Sets seed
        data = data_utils.generate_dataset(self.n_pixels, self.n_classes, self.n_points)
        (X_train, _), (X_test, _) = data
        X_trainvalid = torch.from_numpy(X_train).unsqueeze(1).float()
        N = X_trainvalid.size(0)
        Nvalid = int(N * 0.2)  # Keeps 20% for validation
        X_valid = SimpleShapes.normalize(X_trainvalid[:Nvalid, ...])
        X_train = SimpleShapes.normalize(X_trainvalid[Nvalid:, ...])
        X_test = SimpleShapes.normalize(torch.from_numpy(X_test).unsqueeze(1).float())
        return X_train, X_valid, X_test


class Single(Dataset):
    """Contains x1 transformed with parameters. 
    Total number of samples == x1 transformed
    """

    def __init__(self, X, params):
        self.X = X
        self.params = params

    def __len__(self):
        return self.X.shape[0] * len(self.params)

    @staticmethod
    def collate(batch):
        """Used for dataloader"""
        X1 = torch.stack([item[0] for item in batch])
        X2 = torch.stack([item[1] for item in batch])
        params = [item[2] for item in batch]
        return X1, X2, params

    def get_x_idx(self, idx):
        """Returns the idx of the original image x."""
        return idx // len(self.params)

    def get_x1(self, idx, x_idx):
        x = self.X[x_idx]

        p = len(self.params)
        x1_params_idx = idx % p
        x1_params = self.params[x1_params_idx]

        x1 = transformations.transform(x, x1_params)
        return x1, x1_params

    def __getitem__(self, idx):
        x_idx = self.get_x_idx(idx)
        x1, x1_params = self.get_x1(idx, x_idx)
        x2 = self.X[x_idx]
        return x1, x2, x1_params


class Pairs(Dataset):
    """Contains x1, x2, and transformation params.

    Total of n_samples * num_params^2 pairs:
    (x0, t0) => x1
        (x1, t0) => x2
    (x0, t0) => x1
        (x1, t1) => x2

    Args:
        X (original images): [n_samples, n_pixels, n_pixels]
        params (list of transformations.Params): parameters for transformations
    """

    def __init__(self, X, params):
        self.X = X
        self.params = params

    def __len__(self):
        return self.X.shape[0] * (len(self.params) ** 2)

    @staticmethod
    def collate(batch):
        """Used for dataloader"""
        X1 = torch.stack([item[0] for item in batch])
        X2 = torch.stack([item[1] for item in batch])
        params = [item[2] for item in batch]
        return X1, X2, params

    def get_x_idx(self, idx):
        """Returns the idx of the original image x."""
        return idx // (len(self.params) ** 2)

    def get_x1(self, idx, x_idx):
        x = self.X[x_idx]

        p = len(self.params)
        x1_params_idx = (idx - (x_idx) * p * p) // p
        x1_params = self.params[x1_params_idx]

        x1 = transformations.transform(x, x1_params)
        return x1

    def get_x2_params(self, idx, x_idx):
        p = len(self.params)
        x1_params_idx = (idx - (x_idx) * p * p) // p
        x2_params_idx = idx - ((x_idx * p * p) + (x1_params_idx * p))
        return self.params[x2_params_idx]

    def __getitem__(self, idx):
        x_idx = self.get_x_idx(idx)
        x1 = self.get_x1(idx, x_idx)
        x2_params = self.get_x2_params(idx, x_idx)
        x2 = transformations.transform(x1, x2_params)
        x1, x2 = x1, x2
        return x1, x2, x2_params


class ShapeNet(AbstractDataset):
    pass


class ShapeNetIterator(Dataset):
    """ShapeNet Iterator"""

    def __init__(self, V, transform=None):
        self.V = V
        self.preprocess = transforms.Compose(
            [
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.V[0])

    def __getitem__(self, idx):
        return tuple([self.preprocess(self.V[v][idx]) for v in range(len(self.V))])
