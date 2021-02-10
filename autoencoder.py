"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import copy
import torch
import json
import os
import random
import numpy as np
import models
import latent_operators
import plot
from datasets import datasets, transformations


class AutoEncoder:
    """Trains an autoencoder on rotated shapes.

    Args:
        data (AbstractDataset): contains train and test loaders with transformation params
        z_dim (int): dimension of latent space
        seed (int): for random number generation
        translation (bool): if true, uses an offset identity matrix for rotation
        shift_x (bool): use shift values instead of angles in supervision.

    """

    def __init__(
        self,
        data,
        z_dim=700,
        seed=0,
        encoder_type="Linear",
        decoder_type="Linear",
        latent_operator_name=None,
        device="cpu",
        learning_rate=0.0005,
        n_epochs=5,
    ):

        self.z_dim = z_dim

        self.seed = seed
        self.set_seed()

        self.data = data
        self.device = device

        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

        self.encoder = getattr(models, encoder_type + "Encoder")(
            self.data.n_pixels, self.data.n_channels, z_dim
        ).to(self.device)
        self.decoder = getattr(models, decoder_type + "Decoder")(
            self.data.n_pixels, self.data.n_channels, z_dim
        ).to(self.device)

        self.encoder_best_valid = self.encoder
        self.decoder_best_valid = self.decoder

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.transformation_param_name = self.get_transformation_param_name()
        # function used for latent transformation
        self.use_latent_op = False if latent_operator_name is None else True
        self.latent_operator_name = latent_operator_name
        self.latent_operator = self.get_latent_operator(latent_operator_name)

        self.train_losses = []
        self.valid_losses = []
        self.final_test_loss = None

    def __repr__(self):
        model = {
            "encoder_type": self.encoder_type,
            "decoder_type": self.decoder_type,
            "z_dim": self.z_dim,
            "latent_operator": self.latent_operator_name,
            "batch_size": self.data.batch_size,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "data": str(self.data),
        }
        return json.dumps(model)

    def save(self, path, indices=None):
        os.makedirs(path, exist_ok=True)
        self.save_model_configs(path)
        self.save_models(path)

        self.save_losses(path)
        self.save_plots(path)

    def save_model_configs(self, path):
        model_configs_str = self.__repr__()
        model_configs = json.loads(model_configs_str)

        file_path = os.path.join(path, "model_configs.json")

        with open(file_path, "w") as outfile:
            json.dump(model_configs, outfile)

    def save_models(self, path):
        encoder_path = os.path.join(path, "encoder.pt")
        torch.save(self.encoder.state_dict(), encoder_path)

        decoder_path = os.path.join(path, "decoder.pt")
        torch.save(self.decoder.state_dict(), decoder_path)

    def load_models(self, path, device="cpu"):
        self.encoder.load_state_dict(
            torch.load(os.path.join(path, "encoder.pt"), map_location=device)
        )
        self.decoder.load_state_dict(
            torch.load(os.path.join(path, "decoder.pt"), map_location=device)
        )

    def save_losses(self, path):
        file_path = os.path.join(path, "train_losses.npy")
        np.save(file_path, self.train_losses)

        file_path = os.path.join(path, "valid_losses.npy")
        np.save(file_path, self.valid_losses)

        file_path = os.path.join(path, "test_loss.npy")
        np.save(file_path, self.final_test_loss)

    def save_plots(self, path):

        for train_set in [True, False]:
            set_name = "train" if train_set else "test"

            x1_plot_path = os.path.join(path, f"x1_{set_name}_reconstructions")
            self.plot_x1_reconstructions(save_name=x1_plot_path, train_set=train_set)

            # store x2 reconstructions only when using supervised latent operator
            if self.use_latent_op:
                x2_plot_path = os.path.join(path, f"x2_{set_name}_reconstructions")
                self.plot_x2_reconstructions(
                    save_name=x2_plot_path, train_set=train_set
                )

            transformation_name = (
                "translations"
                if self.transformation_param_name != "angle"
                else "rotations"
            )
            multiple_rotations_path = os.path.join(
                path, f"x_{set_name}_{transformation_name}"
            )
            self.plot_multiple_rotations(
                save_name=multiple_rotations_path, train_set=train_set
            )

    def save_best_validation(self, path, indices=None):
        self.encoder = self.encoder_best_valid
        self.decoder = self.decoder_best_valid
        self.save(path, indices=indices)

    def set_seed(self):
        """Sets seed for random number generation"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        # Generate Dataset
        torch.autograd.set_detect_anomaly(True)

    def get_transformation_param_name(self):
        """Returns the parameter used for transformation"""
        if self.data.n_rotations > 1:
            return "angle"
        elif self.data.n_x_translations > 1:
            return "shift_x"
        elif self.data.n_y_translations > 1:
            return "shift_y"
        else:
            raise ValueError("No transformation found")

    def get_latent_operator(self, name):
        """Returns function to performance transformation based name"""
        if name is None:
            return None
        latent_operator = getattr(latent_operators, name)
        return latent_operator(self.n_transformations, self.device)

    @property
    def n_transformations(self):
        if self.data.n_rotations > 1:
            return self.data.n_rotations
        elif self.data.n_x_translations > 1:
            return self.data.n_x_translations
        elif self.data.n_y_translations > 1:
            return self.data.n_y_translations
        else:
            raise ValueError("No transformation found")

    def train(self, loss_func, stop_early=False, log_frequency=None):
        self.encoder.train().to(self.device)
        self.decoder.train().to(self.device)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        if log_frequency is None:
            log_frequency = self.set_log_frequency()

        for epoch in range(self.n_epochs):

            running_loss = 0.0
            print(f"Epoch {epoch}")
            self.log_train_val_loss(loss_func)
            for i, (x1, x2, params) in enumerate(self.data.train_loader):
                print(f"Training batch {i}", end="\r")
                x1 = x1.to(device=self.device)
                x2 = x2.to(device=self.device)
                angles = self.get_angles(params)
                angles = angles.to(device=self.device)

                optimizer.zero_grad()
                loss = loss_func(x1, x2, angles)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % log_frequency == (log_frequency - 1):
                    print(f"Running loss: {running_loss / log_frequency:0.3e}")
                    running_loss = 0.0
                    if stop_early:
                        return None
        train_loss, valid_loss = self.log_train_val_loss(loss_func)
        self.copy_models_validation(valid_loss)
        # test loss per sample (using batch size 1)
        self.final_test_loss = self.compute_total_loss(
            self.data.test_loader_batch_1, loss_func
        )
        print(f"Test Loss: {self.final_test_loss:0.3e}")

    def set_log_frequency(self):
        frequency = len(self.data.train_loader) // 10
        return frequency

    def copy_models_validation(self, valid_loss):
        """Copies models with best validation"""
        if valid_loss < np.min(self.valid_losses):
            self.encoder_best_valid = copy.deepcopy(self.encoder)
            self.decoder_best_valid = copy.deepcopy(self.decoder)

    def log_train_val_loss(self, loss_func, show_print=True):
        train_loss = self.compute_total_loss(self.data.train_loader, loss_func)
        valid_loss = self.compute_total_loss(self.data.valid_loader, loss_func)
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        if show_print:
            print(f"Total loss train: {train_loss:0.3e} validation: {valid_loss:0.3e}")
        return train_loss, valid_loss

    def compute_total_loss(self, loader, loss_func):
        self.encoder.eval()
        self.decoder.eval()

        losses = []
        with torch.no_grad():
            for x1, x2, params in loader:
                x1 = x1.to(device=self.device)
                x2 = x2.to(device=self.device)
                angles = self.get_angles(params)
                angles = angles.to(device=self.device)
                losses.append(loss_func(x1, x2, angles).cpu())
        mean_loss = torch.stack(losses).mean()

        self.encoder.train()
        self.decoder.train()
        return mean_loss

    def reconstruction_mse_x1(self, x1, x2, angles):
        """Computes MSE x1 reconstruction loss"""
        criterion = torch.nn.MSELoss()
        z = self.encoder(x1)
        x1_reconstruction = self.decoder(z)
        loss = criterion(x1_reconstruction, x1)
        return loss

    def reconstruction_mse_transformed_z1(self, x1, x2, angles):
        """Computes reconstruction MSE of x1 from z1 + x2 from transformed(z1)"""
        criterion = torch.nn.MSELoss()
        z = self.encoder(x1)
        x1_reconstruction = self.decoder(z)
        x1_reconstruction_loss = criterion(x1_reconstruction, x1)
        z_transformed = self.latent_operator(z, angles)
        x2_reconstruction_loss = criterion(self.decoder(z_transformed), x2)

        loss = x1_reconstruction_loss + x2_reconstruction_loss

        return loss

    def reconstruction_mse_frozen_z1(self, x1, x2, angles):
        """Reconstruction loss of x2 from x1 without transformations"""
        criterion = torch.nn.MSELoss()
        z = self.encoder(x1)
        x2_reconstruction = self.decoder(z)
        loss = criterion(x2_reconstruction, x2)
        return loss

    def compute_mean_loss(self, loss_func, data_loader):
        """Computes RMSE based on given loss function."""
        self.encoder.eval().cpu()
        self.decoder.eval().cpu()

        losses = []
        for x1, x2, params in data_loader:
            angles = self.get_angles(params)
            losses.append(loss_func(x1, x2, angles).cpu())
        mean_loss = torch.stack(losses).mean()
        return mean_loss

    def get_angles(self, params):
        """Returns tensor of angles for translations in x or rotations."""
        param_name = self.transformation_param_name
        if param_name in ("shift_x", "shift_y"):
            angles = torch.tensor(
                [
                    transformations.shift_to_angle(
                        getattr(p, param_name), self.n_transformations,
                    )
                    for p in params
                ]
            )
        else:
            angles = torch.tensor([p.angle for p in params])
        return angles

    def run(self, log_frequency=None, stop_early=False):
        """Runs experiment for autoencoder reconstruction.

        Args:
            log_frequency (int): number of batches after which to print loss
            stop_early (bool): stop after a single log_frequency number of batches.
                Useful for testing  without waiting for long training.
        """
        if self.latent_operator_name is None:
            loss_func = self.reconstruction_mse_x1
        elif self.latent_operator_name in ["ShiftOperator", "DisentangledRotation"]:
            loss_func = self.reconstruction_mse_transformed_z1
        # TODO: what is frozen_rotation?
        elif self.latent_operator_name == "frozen_rotation":
            loss_func = self.reconstruction_mse_frozen_z1
        else:
            raise ValueError(
                f"transformation type {self.transformation_type} not supported"
            )
        self.train(
            loss_func, log_frequency=log_frequency, stop_early=stop_early,
        )

    def reconstruct_x1(self, x1):
        """Reconstructs x1 using model"""
        self.encoder.eval().cpu()
        self.decoder.eval().cpu()

        with torch.no_grad():
            z = self.encoder(x1)
            y = self.decoder(z)
        return y

    def reconstruct_transformed_x1(self, x1, param):
        """Reconstructs x1 transformed using model"""
        self.encoder.eval().cpu()
        self.decoder.eval().cpu()

        with torch.no_grad():
            x_transformed = transformations.transform(x1.squeeze(0), param)
            z = self.encoder(x_transformed.unsqueeze(0))
            y = self.decoder(z)
        return y

    def reconstruct_x2(self, x1, param):
        """Reconstructs x2 using model and latent transformation"""
        self.encoder.eval().cpu()
        self.decoder.eval().cpu()

        with torch.no_grad():
            z = self.encoder(x1)
            angle = self.get_angles([param]).unsqueeze(0)
            z_transformed = self.latent_operator(z, angle)
            x2 = self.decoder(z_transformed)
        return x2

    def plot_x1_reconstructions(self, indices=None, train_set=False, save_name=None):
        """Plots x1 autoencoder reconstruction from z1.

        Args:
            pairs (datasets.Pairs): contains x1, x2, and params.
            model (function): callable f(x1) = x1_reconstruction
            indices (list of ints): indices for samples to plot
            train_set (bool): if true title is plotted with train otherwise test.
            save_name (str): indicates path where images should be saved. 
        """
        pairs = self.data.X_train if train_set else self.data.X_test
        if indices is None:
            indices = random.sample(range(len(pairs)), k=4)
        plot.plot_x1_reconstructions(
            pairs, self.reconstruct_x1, indices, train_set, save_name
        )

    def plot_x2_reconstructions(self, indices=None, train_set=False, save_name=None):
        """Plots x1, x2 and x2 autoencoder reconstruction from z1 rotated.

        Args:
            pairs (datasets.Pairs): contains x1, x2, and params.
            model (function): callable f(x1) = x1_reconstruction
            indices (list of ints): indices for samples to plot
            train_set (bool): if true title is plotted with train otherwise test.
            save_name (str): indicates path where images should be saved. 
        """
        pairs = self.data.X_train if train_set else self.data.X_test
        if indices is None:
            indices = random.sample(range(len(pairs)), k=4)
        plot.plot_x2_reconstructions(
            pairs, self.reconstruct_x2, indices, train_set, save_name
        )

    def plot_multiple_rotations(self, indices=None, train_set=False, save_name=None):
        """Plots all rotated reconstructions for given samples"""
        if indices is None:
            n_samples = min(len(self.data.X_orig_train), len(self.data.X_orig_test))
            indices = np.random.randint(low=0, high=n_samples, size=5)
        X = (
            self.data.X_orig_train[indices]
            if train_set
            else self.data.X_orig_test[indices]
        ).float()
        title = (
            "Translations" if self.transformation_param_name != "angle" else "Rotations"
        )
        plot.plot_rotations(
            X,
            self,
            self.n_transformations,
            title,
            save_name=save_name,
            param_name=self.transformation_param_name,
            use_latent_op=self.use_latent_op,
        )


def load_data(configs, path):
    data_configs = json.loads(configs["data"])

    if "shapes" and "2k-classes" in path:
        data = datasets.SimpleShapes(
            configs["batch_size"],
            n_rotations=data_configs["n_rotations"],
            n_x_translations=data_configs["n_x_translations"],
            n_y_translations=data_configs["n_y_translations"],
            n_classes=2000,
            seed=0,
        )
    elif "mnist" in path:
        data = datasets.ProjectiveMNIST(
            configs["batch_size"],
            n_rotations=data_configs["n_rotations"],
            n_x_translations=data_configs["n_x_translations"],
            n_y_translations=data_configs["n_y_translations"],
            train_set_proportion=0.01,
            valid_set_proportion=0.01,
            test_set_proportion=1.0,
            seed=0,
        )
    else:
        raise ValueError("data not found")
    return data


def load(path):
    with open(os.path.join(path, "model_configs.json")) as f:
        configs = json.load(f)
    data = load_data(configs, path)
    model_type = "CCI" if "cci" in path else "Linear"
    model = AutoEncoder(
        data,
        z_dim=configs["z_dim"],
        latent_operator_name=configs["latent_operator"],
        encoder_type=model_type,
        decoder_type=model_type,
    )
    model.load_models(path)
    return model


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")

    n_epochs = 2
    simple_shapes = datasets.SimpleShapes(16)

    print("Training Autoencder")
    model = AutoEncoder(simple_shapes, device=device, n_epochs=n_epochs)
    model.run()

    print("Training Autoencder with Latent Translation")
    model_with_rotation = AutoEncoder(
        simple_shapes,
        latent_operator_name="ShiftOperator",
        device=device,
        n_epochs=n_epochs,
    )
    model_with_rotation.run()
