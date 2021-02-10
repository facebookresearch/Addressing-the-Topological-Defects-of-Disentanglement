"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import matplotlib.pyplot as plt
from datasets import transformations
import torch
import numpy as np 

def plot_x2_reconstructions(
    pairs, model, indices, train_set, save_name,
):
    """
    Plots sample x2 reconstructions based on indices
    
    Args:
        pairs (datasets.Pairs): contains x1, x2, and params.
        model (function): callable f(x1) = x1_reconstruction
        indices (list of ints): indices for samples to plot
        train_set (bool): if true title is plotted with train otherwise test.
        save_name (str): indicates path where images should be saved. 
    """
    title = "Training Reconstruction" if train_set else "Test Reconstruction"
    fig, axs = plt.subplots(len(indices), 3, figsize=(6, 12))
    fig.suptitle(title, fontsize=16)
    for i, sample_idx in enumerate(indices):
        x1, x2, params = pairs[sample_idx]
        n_pixels = x1.shape[1]

        try:
            # for weakly supervised autoencoder
            x2_reconstruction = model(x1.unsqueeze(0), x2.unsqueeze(0), params)
        except TypeError:
            # for real autoencoder
            x2_reconstruction = model(x1.unsqueeze(0), params)

        axs[i][0].imshow(x1.squeeze())
        axs[i][0].set_title("x1")

        axs[i][1].imshow(x2.squeeze())
        axs[i][1].set_title("x2")

        axs[i][2].imshow(
            x2_reconstruction.cpu().detach().numpy().reshape(n_pixels, n_pixels)
        )
        axs[i][2].set_title("x2 from tranformed z1")
        
    if save_name:
        plt.savefig(f"{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_x1_reconstructions(pairs, model, indices, train_set, save_name):
    """
    Plots sample x2 reconstructions based on indices
    
    Args:
        pairs (datasets.Pairs): contains x1, x2, and params.
        model (function): callable f(x1) = x1_reconstruction
        indices (list of ints): indices for samples to plot
        train_set (bool): if true title is plotted with train otherwise test.
        save_name (str): indicates path where images should be saved. 
    """
    title = "Training Reconstructions" if train_set else "Test Reconstructions"
    fig, axs = plt.subplots(len(indices), 2, figsize=(5, 12))
    fig.suptitle(title, fontsize=16)

    for i, sample_idx in enumerate(indices):
        x1, x2, params = pairs[sample_idx]
        n_pixels = x1.shape[1]

        x1_reconstruction = model(x1.unsqueeze(0)).cpu().detach().numpy()

        axs[i][0].imshow(x1.squeeze())
        axs[i][0].set_title("x1")

        axs[i][1].imshow(x1_reconstruction.reshape(n_pixels, n_pixels))
        axs[i][1].set_title("x1 reconstruction")

    if save_name:
        plt.savefig(f"{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_rotations(
    X,
    model,
    n_transformations,
    title,
    save_name=None,
    param_name="angle",
    use_latent_op=True,
):
    """Plots all rotated reconstructions for given samples"""
    font_size = 18
    degree_sign = "\N{DEGREE SIGN}"
    n_samples = X.shape[0]

    fig, axs = plt.subplots(n_samples, n_transformations + 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16)

    for sample_i, x1 in enumerate(X):
        axs[sample_i, 0].imshow(x1.squeeze())
        axs[sample_i, 0].set_title("original", fontsize=font_size)
        axs[sample_i, 0].set_xticks([])
        axs[sample_i, 0].set_yticks([])

        transformation_params = get_all_transformations(param_name, n_transformations)

        for i, param in enumerate(transformation_params):
            if use_latent_op:
                x2_reconstruction = model.reconstruct_x2(x1.unsqueeze(1), param)
            else:
                x2_reconstruction = model.reconstruct_transformed_x1(
                    x1.unsqueeze(1), param
                )
            axs[sample_i, i + 1].imshow(x2_reconstruction.squeeze())
            if param_name == "angle":
                axs[sample_i, i + 1].set_title(
                    f"{param.angle:0.0f}{degree_sign}", fontsize=font_size
                )
            axs[sample_i, i + 1].set_xticks([])
            axs[sample_i, i + 1].set_yticks([])

    if save_name:
        plt.savefig(save_name, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_transformations_complex(
    X,
    model,
    title,
    save_name=None,
    param_name="angle",
    supervised=False,
):
    """Plots all rotated reconstructions for given samples"""
    font_size = 18
    degree_sign = "\N{DEGREE SIGN}"
    n_samples = X.shape[0]
    transformation_params = transformations.get_transform_params(model.data.n_rotations, 
                model.data.n_x_translations, model.data.n_y_translations, (1.0, ))
    n_transformations = len([i for i in transformation_params])
    fig, axs = plt.subplots(n_samples, n_transformations + 1, figsize=(16, int(12/5.*len(X))))

    for sample_i, x1 in enumerate(X):
        axs[sample_i, 0].imshow(x1.squeeze())
        axs[sample_i, 0].set_title("original", fontsize=font_size)
        axs[sample_i, 0].set_xticks([])
        axs[sample_i, 0].set_yticks([])

        x1 = x1.to(model.device)
        z1 = model.encoder(x1)

        transformation_params = transformations.get_transform_params(model.data.n_rotations, 
                model.data.n_x_translations, model.data.n_y_translations, (1.0, ))
        for i, param in enumerate(transformation_params):
            shifts = torch.LongTensor([[i]])
            if supervised:
                z_transformed = model.transform(z1, [shifts])
            else:
                z_transformed = model.transform(z1, torch.LongTensor([[i]]))
            x2_reconstruction = model.decoder(z_transformed).detach().cpu().numpy()

            axs[sample_i, i + 1].imshow(x2_reconstruction.squeeze())
            if param_name == "angle":
                axs[sample_i, i + 1].set_title(
                    f"{param.angle:0.0f}{degree_sign}", fontsize=font_size
                )
            elif param_name == "tx":
                axs[sample_i, i + 1].set_title(f"{param.shift_x:0.0f}", fontsize=font_size)
            elif param_name == 'ty':
                axs[sample_i, i + 1].set_title(f"{param.shift_y:0.0f}", fontsize=font_size)
            else:
                axs[sample_i, i + 1].set_title(f"{param.shift_x:0.0f},{param.shift_y:0.0f}",
                                fontsize=font_size)

            axs[sample_i, i + 1].set_xticks([])
            axs[sample_i, i + 1].set_yticks([])

    if save_name:
        plt.savefig(save_name, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()

def get_all_transformations(param_name, n_transformations):
    if param_name == "angle":
        return transformations.get_transform_params(n_transformations, 0, 0, (1.0,))
    elif param_name == "shift_x":
        return transformations.get_transform_params(0, n_transformations, 0, (1.0,))
    elif param_name == "shift_y":
        return transformations.get_transform_params(0, 0, n_transformations, (1.0,))

def plot_rotations_translations(X, model, n_transformations, n_rot, n_x, n_y, save_name=None):
    degree_sign = "\N{DEGREE SIGN}"
    n_samples = X.shape[0]

    fig, axs = plt.subplots(n_samples, n_transformations + 2, figsize=(16, int(12/5.*len(X))))

    for sample_i, x1 in enumerate(X):
        axs[sample_i, 0].imshow(x1.squeeze())
        axs[sample_i, 0].set_title("original", fontsize=16)
        axs[sample_i, 0].set_xticks([])
        axs[sample_i, 0].set_yticks([])
        x1 = x1.to(model.device)
        transformation_params = [t for t in transformations.get_transform_params(n_rot, n_x, n_y, (1.0, ))]
        z = model.encoder(x1)
        angle = None
        shift_x = None
        shift_y = None
        
        t_list = []
        i = 0
        for _, t in enumerate(range(n_transformations+1)):
            j = np.random.randint(len(transformation_params))
            param = transformation_params[j]
            
            if not t in t_list:
                shifts = model.return_shifts([param])
                z_transformed = model.transform(z, shifts)
                x2_reconstruction = model.decoder(z_transformed).detach().cpu().numpy()

                axs[sample_i, i + 1].imshow(x2_reconstruction.squeeze())
                axs[sample_i, i + 1].set_title(f"{param.angle:0.0f}{degree_sign}\n{param.shift_x:0.0f},{param.shift_y:0.0f}", fontsize=16)
                axs[sample_i, i + 1].set_xticks([])
                axs[sample_i, i + 1].set_yticks([])
                angle = param.angle
                shift_x = param.shift_x
                shift_y = param.shift_y
                i += 1
            if i+1 >= n_transformations + 2:
                break
    if save_name:
        plt.savefig(save_name, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()