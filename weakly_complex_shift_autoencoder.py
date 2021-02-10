"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import models
import latent_operators
from datasets import datasets
from datasets.data_utils import x_to_image
import plot
import pdb
import os
import shutil

eps = 1e-20


class WeaklyComplexAutoEncoder:
    """Trains a weakly supervised shift operator.

    Args:
        data (AbstractDataset): contains train and test loaders with angles
        z_dim (int): dimension of latent space
        seed (int): for random number generation
        translation (bool): if true, uses an offset identity matrix for rotation

    """

    def __init__(
        self,
        data,
        z_dim=405,
        seed=0,
        encoder_type="ComplexLinear",
        decoder_type="ComplexLinear",
        transformation_type=None,
        device="cpu",
        temperature=1.0,
        output_directory="output",
        save_name="",
        use_softmax=1,
        n_rotations = 0,
        n_x_translations = 0,
        n_y_translations = 0,
        scaling_factors = (1, )
    ):

        self.z_dim = z_dim

        self.seed = seed
        self.set_seed()

        self.data = data
        self.device = device

        self.encoder = getattr(models, encoder_type + "Encoder")(
            self.data.n_pixels, self.data.n_channels, z_dim
        ).to(self.device)
        self.decoder = getattr(models, decoder_type + "Decoder")(
        self.data.n_pixels, self.data.n_channels, z_dim
        ).to(self.device)

        cardinals = [
            n_rotations + 1,
            n_x_translations + 1,
            n_y_translations + 1,
            len(scaling_factors),
        ]
        self.cardinals = cardinals 
        # SO FAR, THIS MODEL ONLY WORKS FOR 1 TRANSFO
        # We grab which one with the following line  
        assert (np.array(cardinals) > 1).sum()==1
        for i, cardinal in enumerate(cardinals):
            if cardinal > 1:
                self.K = cardinal
                self.transfo_index = i

        # function used for transformation
        self.use_softmax = use_softmax
        self.transform = self.get_transformation(transformation_type)
        self.temperature = temperature

        self.output_dir = output_directory
        self.save_name = save_name
        self.best_epoch = 0
        self.best_mse = 0

    def set_seed(self):
        """Sets seed for random number generation"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        # Generate Dataset
        torch.autograd.set_detect_anomaly(True)

    def get_transformation(self, name):
        """Returns function to performance transformation based name"""
        if name is None:
            return None
        transformation = getattr(latent_operators, name)
        return transformation(self.cardinals, self.z_dim, self.device, unique_transfo = True)

    def train(self, loss_func, learning_rate, n_epochs, log_frequency):
        self.encoder.train()
        self.decoder.train()

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        train_losses = torch.FloatTensor(n_epochs)
        valid_losses = torch.FloatTensor(n_epochs)
        best_mse = np.inf
        N_pairs = len(self.data.train_loader.dataset)
        for epoch in range(n_epochs):
            epoch_loss = 0
            for i, (x1, x2, angles) in enumerate(self.data.train_loader):
                x1 = x1.to(device=self.device)
                x2 = x2.to(device=self.device)

                optimizer.zero_grad()
                loss = loss_func(x1, x2, angles)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x1.size(0)
            epoch_loss = epoch_loss / N_pairs
            print(f"Epoch {epoch} Train loss: {epoch_loss:0.3e}")

            valid_mse = (
                self.compute_mean_loss(loss_func, self.data.valid_loader)
                .detach()
                .item()
            )

            # train_mse = (
            #     self.compute_mean_loss(loss_func, self.data.train_loader)
            #     .detach()
            #     .item()
            # )
            # train_losses[epoch] = train_mse
            train_losses[epoch] = epoch_loss

            if valid_mse < best_mse:
                self.update_state(mse=valid_mse, epoch=epoch)
                best_mse = valid_mse
                file_name = "checkpoint_{}.pth.tar".format(self.save_name)
                self.save_best_checkpoint(
                    out_dir=self.output_dir,
                    file_name=file_name,
                    optimizer_state_dict=optimizer.state_dict(),
                )

            print(f"Epoch {epoch} validation loss: {valid_mse:0.3e}")
            valid_losses[epoch] = valid_mse
        return train_losses.detach().numpy(), valid_losses.detach().numpy()

    def ifft(self, cps):

        second_dim = cps.size(1)
        K = len(self.transform.translation_matrices[self.transfo_index][0])
        cps_r = cps[..., 0].to(device=self.device)
        cps_i = cps[..., 1].to(device=self.device)
        tr_r = self.transform.translation_matrices[self.transfo_index][0]
        tr_i = self.transform.translation_matrices[self.transfo_index][1]
        alternative = cps_r[:, None, ...] * tr_r - cps_i[:, None, ...] * tr_i
        alternative = alternative.mean(2)  # mean over frequencies
        return alternative

    def reconstruct_x1(self, x1):
        """Reconstructs x1 using model"""
        self.encoder.eval()
        self.decoder.eval()
        x1 = x1.to(device=self.device)

        with torch.no_grad():
            z1 = self.encoder(x1)
            x1_reconstruction_r = self.decoder(z1)
        return x1_reconstruction_r

    def reconstruct_x2(self, x1, x2, param=None):
        """Reconstructs x2 using model and latent transformation"""
        self.encoder.eval()
        self.decoder.eval()
        x1 = x1.to(device=self.device)
        x2 = x2.to(device=self.device)
        batch_size = x1.size(0)

        with torch.no_grad():
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)
            angles_probas = self.compute_angles_probas(x1, x2, z1, z2)
            predicted_angle = angles_probas.detach().argmax(-1, keepdims=True)
            z_transformed = self.transform(z1, predicted_angle)
            x2_reconstruction_r = self.decoder(z_transformed)
        return x2_reconstruction_r

    def plot_multiple_transformations(self, param_name='angle', indices=None, train_set=False, save_name=None):
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
            "Translations" if param_name=='angle' != "angle" else "Rotations"
        )

        plot.plot_transformations_complex(
            X,
            self,
            title,
            save_name=save_name,
            param_name=param_name,
            supervised=False,
        )

    def plot_x1_reconstructions(
        self, indices=[10, 2092, 10299, 13290], train_set=False, save_name=None
    ):
        """Plots x1 autoencoder reconstruction from z1.

            Args:
                pairs (datasets.Pairs): contains x1, x2, and params.
                model (function): callable f(x1) = x1_reconstruction
                indices (list of ints): indices for samples to plot
                train_set (bool): if true title is plotted with train otherwise test.
                save_name (str): indicates path where images should be saved. 
            """
        pairs = self.data.X_train if train_set else self.data.X_test
        plot.plot_x1_reconstructions(
            pairs, self.reconstruct_x1, indices, train_set, save_name
        )

    def plot_x2_reconstructions(
        self, indices=[10, 2092, 10299, 13290], train_set=False, save_name=None
    ):
        """Plots x1, x2 and x2 autoencoder reconstruction from z1 rotated.

        Args:
            pairs (datasets.Pairs): contains x1, x2, and params.
            model (function): callable f(x1) = x1_reconstruction
            indices (list of ints): indices for samples to plot
            train_set (bool): if true title is plotted with train otherwise test.
            save_name (str): indicates path where images should be saved. 
        """
        pairs = self.data.X_train if train_set else self.data.X_test
        plot.plot_x2_reconstructions(
            pairs, self.reconstruct_x2, indices, train_set, save_name
        )

    def compute_angles_probas(self, x1, x2, z1, z2):
        cps = self.computes_cross_power_spectrum(z1[0], z1[1], z2[0], z2[1])
        invfs_alter = self.ifft(cps)
        angles_probas = invfs_alter
        return angles_probas

    def reconstruction_mse_transformed_z1_weak(self, x1, x2, angles, use_argmax=False):
        """Computes reconstruction MSE of x1 from z1 + x2 from transformed(z1), not using ground-truth angles"""
        criterion = torch.nn.MSELoss(reduction="none")
        batch_size = x1.size(0)
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        prod_size = np.prod(x1.size())
        x1_reconstruction_r = self.decoder(z1)
        x1_reconstruction_loss = criterion(x1_reconstruction_r, x1)
        x1_reconstruction_loss = x1_reconstruction_loss.mean()

        # TODO this is not adapted to product of shift operators, it's looking only at the 1st cardinal
        # Transform according to all possible angles, weighted
        angles_probas = self.compute_angles_probas(x1, x2, z1, z2)

        if use_argmax:
            predicted_angle = angles_probas.detach().argmax(
                -1, keepdims=True
            )
            z_transformed = self.transform(z1, predicted_angle)
            x2_reconstruction_r = self.decoder(z_transformed)
            x2_reconstruction_loss = criterion(x2_reconstruction_r, x2)
            x2_reconstruction_loss = x2_reconstruction_loss.mean()
        
        else:
            all_angles = torch.arange(self.K).repeat(1, batch_size).view(-1, 1)
            temp = self.temperature 
            mask = torch.softmax(angles_probas / temp, dim=-1)

            repeat_z1 = (
                z1[0][:, None, :].repeat(1, self.K, 1).view(batch_size * self.K, -1),
                z1[1][:, None, :].repeat(1, self.K, 1).view(batch_size * self.K, -1),
            )

            x2_repeat = (
                x2[:, None, ...]
                .repeat(1, self.K, 1, 1, 1)
                .view(batch_size * self.K, x2.size(1), x2.size(2), x2.size(3))
            )
            z_transformed = self.transform(repeat_z1, all_angles)
            x2_reconstruction_r = self.decoder(z_transformed)
            x2_reconstruction_transformed_loss = (
                criterion(x2_reconstruction_r, x2_repeat)
                .sum((1, 2, 3)) # sums over image dim
                .view(batch_size, -1)
            )
            x2_reconstruction_loss = (mask * x2_reconstruction_transformed_loss).sum() / prod_size

        loss = x1_reconstruction_loss + x2_reconstruction_loss
        return loss

    def computes_cross_power_spectrum(
        self, z_batch_r1, z_batch_i1, z_batch_r2, z_batch_i2
    ):
        """Computes Cross Power spectrum (no FFT) """
        batch_size = z_batch_r1.size(0)
        z1z2_batch_r = (
            z_batch_r1 * z_batch_r2 + z_batch_i1 * z_batch_i2
        )  # recall we use the conjugate of z_batch_2, hence the + here
        z1z2_batch_i = (
            -z_batch_r1 * z_batch_i2 + z_batch_i1 * z_batch_r2
        )  # recall we use the conjugate of z_batch_2, hence the - in front here

        norm_z1z2_batch = ((z1z2_batch_r ** 2 + z1z2_batch_i ** 2)) ** 0.5
        cps_r = z1z2_batch_r / norm_z1z2_batch
        cps_i = z1z2_batch_i / norm_z1z2_batch
        cps = torch.cat([cps_r[..., None], cps_i[..., None]], -1)
        return cps

    def compute_test_loss(self, loss_func, data_loader):
        """Computes RMSE based on given loss function."""
        self.encoder.eval()
        self.decoder.eval()
        losses = []
        N = 0
        with torch.no_grad():
            for i, (x1, x2, angles) in enumerate(data_loader):
                x1 = x1.to(device=self.device)
                x2 = x2.to(device=self.device)
                bs = x1.size(0)
                loss_batch = loss_func(x1, x2, angles, True)*bs
                N += bs
                losses.append(loss_batch)
                
        test_loss = torch.stack(losses).sum() / float(N)
        
        self.encoder.train()
        self.decoder.train()
        return test_loss

    def compute_mean_loss(self, loss_func, data_loader):
        """Computes RMSE based on given loss function."""
        self.encoder.eval()
        self.decoder.eval()
        losses = []
        with torch.no_grad():
            for i, (x1, x2, angles) in enumerate(data_loader):
                x1 = x1.to(device=self.device)
                x2 = x2.to(device=self.device)
                loss_batch = loss_func(x1, x2, angles, True)
                losses.append(loss_batch)
        mean_loss = torch.stack(losses).mean()
        self.encoder.train()
        self.decoder.train()
        return mean_loss

    def run(
        self, learning_rate=0.0005, n_epochs=10, log_frequency=50
        ):
        """Runs experiment for autoencoder reconstruction."""
        loss_func = self.reconstruction_mse_transformed_z1_weak
        train_loss, valid_loss = self.train(
            loss_func, learning_rate, n_epochs, log_frequency
        )

        train_mse = self.compute_mean_loss(loss_func, self.data.train_loader)
        print(f"Train MSE: {train_mse}")
        valid_mse = self.compute_mean_loss(loss_func, self.data.valid_loader)
        print(f"Valid MSE: {valid_mse}")
        test_mse = self.compute_test_loss(loss_func, self.data.test_loader_batch_100)
        print(f"Test MSE: {test_mse}")
        return train_loss, valid_loss, train_mse, valid_mse, test_mse

    def update_state(self, mse, epoch):
        self.best_mse = mse
        self.best_epoch = epoch

    def load_model(self, path_to_checkpoint):
        checkpoint = torch.load(path_to_checkpoint)
        self.best_epoch = checkpoint["best_epoch"]
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.best_mse = checkpoint["best_mse"]
        return checkpoint["best_mse"], checkpoint["best_epoch"]

    def get_current_state(self):
        return {
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "best_epoch": self.best_epoch,
            "best_mse": self.best_mse,
        }

    def save_best_checkpoint(self, out_dir, file_name, optimizer_state_dict):
        """
        :param file_name: filename to save checkpoint in.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        best_path = os.path.join(out_dir, "best_" + file_name)
        torch.save(state, best_path)
