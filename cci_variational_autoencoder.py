"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
"""Implements CCI VAE
https://arxiv.org/abs/1804.03599
"""

import torch
import os
import numpy as np
import models
import json
import plot
import copy
import random
from datasets import datasets, transformations
from datasets.data_utils import x_to_image
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt


class CCIVariationalAutoEncoder:
    """Trains an autoencoder on rotated shapes.

    Args:
        data (AbstractDataset): contains train and test loaders with angles
        model (CCIVAE model): contains forward funtion with encoder / decoder
        beta (float): beta in beta-VAE model
        c_max (float): maximum value for controlled capacity parameter in CCI VAE.
        z_dim (int): dimension of latent space
        seed (int): for random number generation
        translation (bool): if true, uses an offset identity matrix for rotation

    """

    def __init__(
        self,
        data,
        model=models.CCIVAE,
        beta=1000.0,
        c_max=36.0,
        z_dim=30,
        seed=0,
        device="cpu",
        learning_rate=0.0005,
        n_epochs=5,
        distribution="gaussian",
    ):
        self.beta, self.c_max = beta, c_max
        self.c = 0.0
        self.z_dim = z_dim

        self.data = data
        self.device = device
        self.model_cls = model

        self.model = model(
            self.data.n_pixels, self.data.n_channels, z_dim, distribution=distribution
        )
        self.model.to(device=device)

        self.model_best_valid = self.model

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.distribution = distribution

        self.seed = seed
        self.set_seed()

        self.train_losses = []
        self.kl_losses = []
        self.reconstruction_losses = []

        self.valid_losses = []
        self.final_test_loss = None

    def __repr__(self):
        model = {
            "model_class": str(self.model_cls),
            "beta": self.beta,
            "c_max": self.c_max,
            "distribution": self.distribution,
            "z_dim": self.z_dim,
            "batch_size": self.data.batch_size,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "data": str(self.data),
        }
        return json.dumps(model)

    def set_seed(self):
        """Sets seed for random number generation"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        # Generate Dataset
        torch.autograd.set_detect_anomaly(True)

    def compute_loss(self, x1):
        """Loss for controlled capacity beta vae (CCI VAE)
        https://arxiv.org/abs/1804.03599
        """
        if self.distribution == "gaussian":
            criterion = torch.nn.MSELoss(reduction="sum")
        elif self.distribution == "bernoulli":
            criterion = torch.nn.BCELoss(reduction="sum")
        else:
            raise ValueError(f"distribution {self.distribution} not supported")

        # assuming a Gaussian Distribution
        out, mu, log_var = self.model(x1)
        reconstruction_loss = criterion(out, x1)

        # https://arxiv.org/abs/1312.6114
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_divergence = (
            -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean(dim=0)
        ).sum()

        return reconstruction_loss, kl_divergence

    def train(self, stop_early=False, log_frequency=None, track_losses=True):
        """Trains controlled capacity beta vae (CCI VAE)
        https://arxiv.org/abs/1804.03599

        Learning rate used in the paper is 5e-4

        If verbose is False, previous loss print is overridden
        If stop_early is True, training stops after first logged loss. 
        This is useful for testing.
        """
        self.model.train().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        c_step_size = (self.c_max - self.c) / self.n_epochs

        if log_frequency is None:
            log_frequency = self.set_log_frequency()

        for epoch in range(self.n_epochs):
            running_loss = 0.0
            print(f"Epoch {epoch}")
            if track_losses:
                self.log_train_val_loss()
            running_loss = 0.0
            running_reconstruction_loss, running_kl_divergence = 0.0, 0.0
            # update controlled capacity parameter
            self.c += c_step_size
            for i, (x1, _, _) in enumerate(self.data.train_loader):
                x1 = x1.to(device=self.device)

                optimizer.zero_grad()
                reconstruction_loss, kl_divergence = self.compute_loss(x1)

                loss = reconstruction_loss + self.beta * (kl_divergence - self.c).abs()

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_reconstruction_loss += (
                    reconstruction_loss.cpu().detach().numpy()
                )
                running_kl_divergence += kl_divergence.cpu().detach().numpy()

                if i % log_frequency == (log_frequency - 1):
                    normalized_loss = running_loss / log_frequency
                    normalized_reconstruction_loss = (
                        running_reconstruction_loss / log_frequency
                    )
                    normalized_kl_divergence = running_kl_divergence / log_frequency
                    print(f"Running Total Loss: {normalized_loss:0.3e}")
                    print(
                        f"Running Reconstruction Loss: {normalized_reconstruction_loss:0.3e}"
                        f" KL Divergence: {normalized_kl_divergence:0.3e}"
                    )
                    self.kl_losses.append(normalized_kl_divergence)
                    self.reconstruction_losses.append(normalized_reconstruction_loss)

                    running_loss = 0.0
                    running_reconstruction_loss = 0.0
                    running_kl_divergence = 0.0
                    if stop_early:
                        return None

        if track_losses:
            train_loss, valid_loss = self.log_train_val_loss()
            self.copy_models_validation(valid_loss)
            # compute test loss per sample
            self.final_test_loss = self.compute_total_loss(
                self.data.test_loader_batch_1
            )
            print(f"Test Loss: {self.final_test_loss:0.3e}")

    def set_log_frequency(self):
        frequency = len(self.data.train_loader) // 10
        return frequency

    def copy_models_validation(self, valid_loss):
        """Copies models with best validation"""
        if valid_loss < np.min(self.valid_losses):
            self.model_vest_valid = copy.deepcopy(self.model)

    def log_train_val_loss(self, show_print=True):
        train_loss = self.compute_total_loss(self.data.train_loader)
        valid_loss = self.compute_total_loss(self.data.valid_loader)
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        if show_print:
            print(f"Total loss train: {train_loss:0.3e} validation: {valid_loss:0.3e}")
        return train_loss, valid_loss

    def compute_total_loss(self, loader):
        """Computes total average loss on given loader"""
        self.model.eval()
        losses = []
        with torch.no_grad():
            for x1, x2, params in loader:
                x1 = x1.to(device=self.device)
                reconstruction_loss, kl_divergence = self.compute_loss(x1)
                loss = reconstruction_loss + self.beta * (kl_divergence - self.c).abs()
                losses.append(loss.item())
        mean_loss = np.mean(losses)
        self.model.train()
        return mean_loss

    def reconstruct_x1(self, x1):
        """Reconstructs x1 using model"""
        self.model.eval().cpu()

        with torch.no_grad():
            y, _, _ = self.model(x1)
        return y

    def reconstruct_mean(self, x1):
        self.model.eval().cpu()

        with torch.no_grad():
            _, mu, _ = self.model(x1)
            out = self.model.decoder(mu)
        return out

    def save_best_validation(self, path, indices=None):
        """Saves results best for model with best validation loss"""
        self.model = self.model_best_valid
        self.save(path, indices=indices)

    def save(self, path, indices=None):
        os.makedirs(path, exist_ok=True)
        self.save_model_configs(path)
        self.save_model(path)

        self.save_losses(path)
        self.save_plots(path)

    def save_model_configs(self, path):
        model_configs_str = self.__repr__()
        model_configs = json.loads(model_configs_str)

        file_path = os.path.join(path, "model_configs.json")

        with open(file_path, "w") as outfile:
            json.dump(model_configs, outfile)

    def load_model(self, path):
        device = torch.device("cpu")
        model = self.model_cls(self.data.n_pixels, self.data.n_channels, self.z_dim)
        model.load_state_dict(torch.load(path, map_location=device))
        self.model = model
        self.model.to(device=device)

    def save_model(self, path):
        full_path = os.path.join(path, "model.pt")
        torch.save(self.model.state_dict(), full_path)

    def save_losses(self, path):
        file_path = os.path.join(path, "kl_divergence.npy")
        np.save(file_path, self.kl_losses)

        file_path = os.path.join(path, "reconstruction_losses.npy")
        np.save(file_path, self.reconstruction_losses)

        file_path = os.path.join(path, "train_losses.npy")
        np.save(file_path, self.train_losses)

        file_path = os.path.join(path, "valid_losses.npy")
        np.save(file_path, self.valid_losses)

        file_path = os.path.join(path, "test_loss.npy")
        np.save(file_path, self.final_test_loss)

    def save_plots(self, path):
        matplotlib.use("Agg")

        for train_set in [True, False]:
            set_name = "train" if train_set else "test"

            x1_plot_path = os.path.join(path, f"x1_{set_name}_reconstructions")
            self.plot_x1_reconstructions(save_name=x1_plot_path, train_set=train_set)

            latent_traversal_path = os.path.join(path, f"x_{set_name}_latent_traversal")
            self.plot_latent_traversal(
                save_name=latent_traversal_path, train_set=train_set
            )

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
            pairs, self.reconstruct_mean, indices, train_set, save_name
        )

    def plot_latent_traversal(
        self,
        indices=None,
        num_std=6.0,
        train_set=True,
        save_name=None,
        fixed_range=True,
    ):
        """Traverses latent space from [mu - 3 * std, mu + 3 * std] for given indices.
        If fixed_range is True, then [-num_std, num_std] is the interval.
        """
        self.model.eval().cpu()

        pairs = self.data.X_train if train_set else self.data.X_test
        if indices is None:
            indices = random.sample(range(len(pairs)), k=3)

        for index in indices:
            sample_save_name = save_name
            if save_name is not None:
                sample_save_name = save_name + "_sample_" + str(index)
            self._plot_latent_traversal_helper(
                pairs, index, num_std, train_set, sample_save_name, fixed_range
            )

    def plot_single_latent_traversal(
        self, index=3, train_set=True, latent_dim=0, save_name=None, num_std=6.0,
    ):
        self.model.eval().cpu()

        pairs = self.data.X_train if train_set else self.data.X_test
        sample_save_name = save_name
        if save_name is not None:
            sample_save_name = save_name + "_sample_" + str(index)

        x1, x2, p = pairs[index]
        title = "Training" if train_set else "Test"

        traversal_path = CCIVariationalAutoEncoder.get_std_path(num_std)
        num_subplots = len(traversal_path) + 1

        fig, axs = plt.subplots(1, num_subplots, figsize=(12, 16))

        axs[0].imshow(x1.squeeze())
        axs[0].set_title(f"{title}: x1, latent {latent_dim}")

        axs[0].set_xticks([])
        axs[0].set_yticks([])

        with torch.no_grad():
            _, mu, log_var = self.model(x1.unsqueeze(0))
            z = mu

        for i, step in enumerate(traversal_path):
            z_shifted = z.clone().cpu().detach()
            z_shifted[0][latent_dim] = step

            with torch.no_grad():
                reconstruction = self.model.decoder(z_shifted)
            axs[i + 1].imshow(reconstruction.squeeze().detach().numpy())

            axs[i + 1].set_xticks([])
            axs[i + 1].set_yticks([])

        fig.tight_layout()

        if save_name:
            # close figure to speed up saving
            plt.savefig(sample_save_name, bbox_inches="tight", dpi=100)
            plt.close(fig)

    @staticmethod
    def get_std_path(num_std):
        """Returns list of std steps.
        [-3, -2, -1, 0, 1, 2, 3]
        """
        step_size = num_std / 3.0

        positive_steps = [i * step_size for i in range(1, 4)]
        negative_steps = sorted(list(-1 * np.array(positive_steps)))
        path = negative_steps + [0] + positive_steps
        return path

    def _plot_latent_traversal_helper(
        self, X, index, num_std, train_set, save_name, fixed_range
    ):
        title = "Training" if train_set else "Test"

        traversal_path = CCIVariationalAutoEncoder.get_std_path(num_std)
        num_subplots = len(traversal_path) + 1

        x1, x2, p = X[index]
        fig, axs = plt.subplots(self.z_dim, num_subplots, figsize=(20, 60))

        for dim in range(self.z_dim):
            axs[dim, 0].imshow(x1.squeeze())
            axs[dim, 0].set_title(f"{title}: x1, latent {dim}")

            axs[dim, 0].set_xticks([])
            axs[dim, 0].set_yticks([])

            with torch.no_grad():
                _, mu, log_var = self.model(x1.unsqueeze(0))
                z = mu

            for i, step in enumerate(traversal_path):
                if not fixed_range:
                    z_shifted = CCIVariationalAutoEncoder.shift_latent(
                        z, dim, step, log_var
                    )
                else:
                    z_shifted = z.clone().cpu().detach()
                    z_shifted[0][dim] = step

                with torch.no_grad():
                    reconstruction = self.model.decoder(z_shifted)
                axs[dim, i + 1].imshow(reconstruction.squeeze().detach().numpy())
                if not fixed_range:
                    axs[dim, i + 1].set_title(f"std {step:.1f}")
                else:
                    axs[dim, i + 1].set_title(f"{step:.1f}")

                axs[dim, i + 1].set_xticks([])
                axs[dim, i + 1].set_yticks([])

        fig.tight_layout()

        if save_name:
            # close figure to speed up saving
            plt.savefig(save_name, bbox_inches="tight", dpi=100)
            plt.close(fig)

    @staticmethod
    def shift_latent(z, dim, num_std, log_var):
        """Shifts latent by num_std along index of latent dimension"""
        std = torch.exp(log_var / 2.0)
        z_shifted = z.clone().cpu().detach()
        z_shifted[0][dim] += num_std * std[0][dim]
        return z_shifted

    def get_latents(self, train_set=False, num_batches=1000):
        """Returns latent representation for random indices"""
        self.model.eval().cpu()

        loader = self.data.train_loader if train_set else self.data.test_loader

        Z = []

        for i, (x1, x2, p) in enumerate(loader):
            z = self.get_latent(x1)
            Z.append(z)
            if i == num_batches:
                break
        Z = torch.cat(Z)
        return Z

    def get_latent(self, x):
        with torch.no_grad():
            _, mu, var = self.model(x)
            z = self.model.reparameterize(mu, var)
        return z

    def compute_latent_variances(self, n_samples=None):
        """Computes variance of latents across transformations of a sample"""
        if n_samples is None:
            n_samples = len(self.data.X_orig_test)

        variances = []

        for i in range(n_samples):
            x1 = self.data.X_orig_test[i]
            self.model.eval().cpu()

            with torch.no_grad():
                sample_latents = []
                for param in self.data.transform_params:
                    x_transformed = transformations.transform(x1, param)
                    _, mu, log_var = self.model(x_transformed.unsqueeze(0))
                    # use mean of latent
                    z = mu
                    sample_latents.append(z)
                sample_latents = torch.cat(sample_latents)
                sample_var = sample_latents.var(dim=0)
                variances.append(sample_var)

        variances = torch.stack(variances).numpy()
        return variances

    def compute_latents_per_shape(self, n_samples=None):
        """Computes variance of latents across transformations of a sample"""
        if n_samples is None:
            n_samples = len(self.data.X_orig_test)

        latents = []

        for i in range(n_samples):
            x1 = self.data.X_orig_test[i]
            self.model.eval().cpu()

            with torch.no_grad():
                sample_latents = []
                for param in self.data.transform_params:
                    x_transformed = transformations.transform(x1, param)
                    _, mu, log_var = self.model(x_transformed.unsqueeze(0))
                    # use mean of latent
                    z = mu
                    sample_latents.append(z)
                sample_latents = torch.cat(sample_latents)

            latents.append(sample_latents)

        latents = torch.stack(latents).numpy()
        return latents

    def pca_ranked_eigenvalues(self, n_samples=None):
        """Returns average of ranked normalized eigenvalues for latents"""
        latents = self.compute_latents_per_shape(n_samples=n_samples)
        n_components = self.data.n_rotations + 1
        aggregate_ranked_normalized_eigenvalues = []

        for latent in latents:
            pca = PCA(n_components=n_components)
            pca.fit(latents[1])
            ranked_normalized_eigenvalues = np.sort(pca.explained_variance_ratio_)[::-1]
            aggregate_ranked_normalized_eigenvalues.append(
                ranked_normalized_eigenvalues
            )

        aggregate_ranked_normalized_eigenvalues = np.stack(
            aggregate_ranked_normalized_eigenvalues
        )
        average_var_explained = np.mean(aggregate_ranked_normalized_eigenvalues, axis=0)
        return average_var_explained


def compute_mutual_info(variances):
    """Variances is a numpy array with shape (n_samples, z_dim)"""
    n = variances.shape[0]
    m_info = np.log(2 * np.pi * variances).sum(0) / (2.0 * n)
    return m_info


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
        data = datasets.ProjectiveSingleDigitMNIST(
            configs["batch_size"],
            n_rotations=data_configs["n_rotations"],
            n_x_translations=data_configs["n_x_translations"],
            n_y_translations=data_configs["n_y_translations"],
            train_set_proportion=0.1,
            valid_set_proportion=0.1,
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
    model = CCIVariationalAutoEncoder(
        data,
        z_dim=configs["z_dim"],
        beta=configs["beta"],
        c_max=configs["c_max"],
        distribution=configs["distribution"],
        learning_rate=configs["learning_rate"],
        n_epochs=configs["n_epochs"],
    )
    model.load_model(os.path.join(path, "model.pt"))
    return model


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")

    n_epochs = 2
    batch_size = 16
    simple_shapes = datasets.SimpleShapes(batch_size)

    vae = CCIVariationalAutoEncoder(
        simple_shapes, beta=0.0, c_max=0.0, device=device, n_epochs=n_epochs
    )
    vae.train()
