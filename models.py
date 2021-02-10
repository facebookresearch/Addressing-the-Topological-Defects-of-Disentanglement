"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from torch import nn
from collections import OrderedDict
from abc import ABC


class ResNetExplorer(nn.Module):
    """
    Loads a pre-trained model and hook on one of its layer 
    """

    def __init__(self, path_to_model="pytorch/vision:v0.6.0", model="resnet152"):
        super().__init__()
        self.pretrained_model = torch.hub.load(path_to_model, model, pretrained=True)

    def create_full_model(self, layer_to_explore, layer_to_explore_size, image_size):

        all_layers = dict(list(self.pretrained_model.named_children()))
        all_keys = list(
            all_layers.keys()
        )  # TODO: I am not sure the order is preserved ?
        max_index = all_keys.index(layer_to_explore)

        ##### ENCODER
        # take all layers up to the one we want to explore for the encoder
        encoder_layers = [
            (all_keys[i], all_layers[layer])
            for i, layer in enumerate(all_layers)
            if i <= max_index
        ]
        layers = OrderedDict()
        for layer in encoder_layers:
            name = layer[0]
            layers[name] = layer[1]

        # create a new model with it (saves time during feed-forward if we take other layers than the last one)
        self.fixed_encoder = nn.Sequential(layers)

        ##### Linear layer to learn the mapping
        self.linear = nn.Linear(layer_to_explore_size, layer_to_explore_size)

        ##### DECODER
        self.decoder = nn.Linear(layer_to_explore_size, image_size)

    def forward(self, x):

        z = self.fixed_encoder(x)
        # feed flattened z to linear
        z_prime = self.linear(z.view(x.size(0), -1))
        x_dec = self.decoder(z_prime)
        # sigmoid to have something between 0 and 1
        x_dec = torch.sigmoid(x_dec)
        # map to image shape
        return x_dec.view(x.size())


class LinearEncoder(nn.Module):
    def __init__(self, n_pixels, n_channels, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(n_pixels ** 2 * n_channels, z_dim, bias=False)

    def forward(self, x):
        out = x.flatten(start_dim=1)
        out = self.fc1(out)
        return out


class LinearDecoder(nn.Module):
    def __init__(self, n_pixels, n_channels, z_dim):
        super().__init__()
        self.n_pixels = n_pixels
        self.n_channels = n_channels
        self.fc1 = nn.Linear(z_dim, n_pixels ** 2 * n_channels, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        out = out.reshape(-1, self.n_channels, self.n_pixels, self.n_pixels)
        return out


class ComplexLinearEncoder(nn.Module):
    def __init__(self, n_pixels, n_channels, z_dim):
        super().__init__()
        self.fc1r = torch.nn.Linear(n_pixels ** 2 * n_channels, z_dim, bias=False)
        self.fc1i = torch.nn.Linear(n_pixels ** 2 * n_channels, z_dim, bias=False)

    def forward(self, x):
        out = x.flatten(start_dim=1)
        outr = self.fc1r(out)
        outi = self.fc1i(out)
        return (outr, outi)


class ComplexLinearDecoder(nn.Module):
    def __init__(self, n_pixels, n_channels, z_dim):
        super().__init__()
        self.n_pixels = n_pixels
        self.n_channels = n_channels
        self.fc1r = nn.Linear(z_dim, n_pixels ** 2 * n_channels, bias=False)
        self.fc1i = nn.Linear(z_dim, n_pixels ** 2 * n_channels, bias=False)

    def forward(self, x):
        r1 = self.fc1r(x[0])
        r2 = -self.fc1i(x[1])
        out_r = r1 + r2
        out_r = out_r.reshape(-1, self.n_channels, self.n_pixels, self.n_pixels)
        return out_r


class CCIEncoder(nn.Module):
    def __init__(self, n_pixels, n_channels, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_pixels, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(n_pixels, n_pixels, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(n_pixels, n_pixels, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(n_pixels, 256, kernel_size=1, stride=1),
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(256, z_dim),
        )

    def forward(self, x):
        out = self.encoder(x)
        return out


class CCIDecoder(nn.Module):
    def __init__(self, n_pixels, n_channels, z_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            Lambda(lambda x: x.view(-1, 256, 1, 1)),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, n_pixels, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(n_pixels, n_channels, 4, 2, 1),
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(32 * 32, n_pixels * n_pixels),
            Lambda(lambda x: x.view(x.size(0), 1, n_pixels, n_pixels)),
        )

    def forward(self, x):
        out = self.decoder(x)
        return out


class NonLinearEncoder(nn.Module):
    def __init__(self, n_pixels, n_chanels, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(n_pixels ** 2, n_pixels // 2)
        self.batch_norm = nn.BatchNorm1d(n_pixels // 2)
        self.fc2 = nn.Linear(n_pixels // 2, z_dim)

    def forward(self, x):
        out = x.flatten(start_dim=1)
        out = self.fc1(out)
        out = self.batch_norm(out)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        return out


class NonLinearDecoder(nn.Module):
    def __init__(self, n_pixels, n_channels, z_dim):
        super().__init__()
        self.n_channels = n_channels
        self.n_pixels = n_pixels
        self.fc1 = nn.Linear(z_dim, (n_pixels ** 2) // 2)
        self.batch_norm = nn.BatchNorm1d((n_pixels ** 2) // 2)
        self.fc2 = nn.Linear((n_pixels ** 2) // 2, n_pixels ** 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        # reshape
        out = self.fc2(out)
        out = torch.relu(out)
        out = out.reshape(-1, self.n_channels, self.n_pixels, self.n_pixels)
        return out


class VAEBase(ABC):
    @staticmethod
    def reparameterize(mu, log_var):
        """Returns z_sample from mu, var"""
        std = torch.exp(log_var / 2)
        # z_sample = torch.normal(mu, std)
        # eps = Variable(torch.randn_like(std))
        eps = torch.randn_like(std)
        z_sample = mu + eps.mul(std)
        return z_sample

    @staticmethod
    def latent_sample(mu, log_var, num_std):
        """Generates sample based on mu, var that's num_std away from mean"""
        std = torch.exp(log_var / 2)
        z_sample = (num_std * std).add(mu)
        return z_sample


class LinearCCIVAE(nn.Module, VAEBase):
    def __init__(self, n_pixels, n_channels, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = LinearEncoder(n_pixels, n_channels, 2 * z_dim)
        self.decoder = LinearDecoder(n_pixels, n_channels, z_dim)

    def forward(self, x):
        z_dist = self.encoder(x)
        mu, log_var = z_dist[:, : self.z_dim], z_dist[:, self.z_dim :]
        # reparameterize
        z_sample = LinearCCIVAE.reparameterize(mu, log_var)
        out = self.decoder(z_sample)
        return out, mu, log_var


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class CCIVAE(nn.Module, VAEBase):
    """Model Architecture from CCI-VAE paper 
    https://arxiv.org/abs/1804.03599
    Encoder:
        4 convolutional layers, each with 32 channels, 4x4 kernels, and a stride of 2. 
       Followed by 2 fully connected layers, each of 256 units

    Latent Space: 20 units (10 for mean, 10 for variance)

    Decoder:
        transpose of encoder with ReLU activations
    """

    def __init__(self, n_pixels, n_channels, z_dim, distribution="gaussian"):
        super().__init__()

        self.z_dim = z_dim
        self.n_channels = n_channels
        self.distribution = distribution
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_pixels, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(n_pixels, n_pixels, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(n_pixels, n_pixels, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(n_pixels, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(256, 2 * z_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            Lambda(lambda x: x.view(-1, 256, 1, 1)),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, n_pixels, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(n_pixels, n_channels, 4, 2, 1),
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.ReLU(),
            nn.Linear(32 * 32, n_pixels * n_pixels),
            Lambda(lambda x: x.view(x.size(0), 1, n_pixels, n_pixels)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z_dist = self.encoder(x)
        mu, log_var = z_dist[:, : self.z_dim], z_dist[:, self.z_dim :]
        # tanh log_var didn't seem to help
        # log_var = torch.tanh(log_var)
        z_sample = CCIVAE.reparameterize(mu, log_var)
        out = self.decoder(z_sample)
        return out, mu, log_var
