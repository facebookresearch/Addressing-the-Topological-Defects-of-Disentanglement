"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

---
Saves model/plots for best validation MSE
"""

import math
import numpy as np
import os
from distutils.dir_util import copy_tree


def save_best_validation_helper(folder, operator):
    min_valid_loss = math.inf
    for sweep in os.listdir(folder):
        if sweep.startswith("best") or sweep.startswith(".DS_Store"):
            continue
        path = os.path.join(folder, sweep, operator)
        try:
            valid_loss = np.min(np.load(os.path.join(path, "valid_losses.npy")))
        except FileNotFoundError:
            print(f"run {sweep} missing for {operator}")
            continue
        if min_valid_loss >= valid_loss:
            min_valid_loss = valid_loss
            destination = os.path.join(folder, "best-validation", operator)
            copy_tree(path, destination)


def save_all_best_validation(parent_folder):
    for experiment in os.listdir(parent_folder):
        experiment_path = os.path.join(parent_folder, experiment)
        if experiment.endswith("-sweep") and "autoencoder" in experiment and "standard" not in experiment:
            save_best_validation_helper(experiment_path, "disentangled-operator")
            save_best_validation_helper(experiment_path, "shift-operator")
        elif experiment.endswith("-sweep") and "standard-autoencoder" in experiment: 
            save_best_validation_helper(experiment_path, "standard-autoencoder")
        elif experiment.endswith("-sweep") and "cci-vae" in experiment:
            save_best_validation_helper(experiment_path, "cci_vae")
            save_best_validation_helper(experiment_path, "beta_vae")
            save_best_validation_helper(experiment_path, "vae")


if __name__ == "__main__":
    user = os.environ["USER"]
    parent_folder = f"/checkpoint/{user}/Equivariance/"
    save_all_best_validation(parent_folder)
