"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
import torch
import sys

sys.path.append("..")
from datasets import datasets
from weakly_complex_shift_autoencoder import WeaklyComplexAutoEncoder
from complex_shift_autoencoder import ComplexAutoEncoder
import sys
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn

use_cuda = True if torch.cuda.is_available() else False

parser = argparse.ArgumentParser(
    description="Fully/Weakly supervised version of shift operator"
)

# General arguments
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--output_directory",
    type=str,
    default="output",
    help="In this directory the models will be "
    "saved. Will be created if doesn't exist.",
)
parser.add_argument("--n_epochs", type=int, default="10", help="Number of epochs.")
parser.add_argument("--lr", type=float, default="0.001", help="Learning rate.")
parser.add_argument("--bs", type=int, default="16", help="Batch size.")

parser.add_argument(
    "--n_rot", type=int, default="9", help="Number of rotations (for the model)."
)
parser.add_argument(
    "--data_n_rot", type=int, default="9", help="Number of rotations (for the data)."
)

parser.add_argument(
    "--n_x",
    type=int,
    default="0",
    help="Number of x translations in x (for the model).",
)
parser.add_argument(
    "--data_n_x",
    type=int,
    default="0",
    help="Number of x translations in x (for the data).",
)
parser.add_argument(
    "--n_y",
    type=int,
    default="0",
    help="Number of y translations in y (for the model).",
)
parser.add_argument(
    "--data_n_y",
    type=int,
    default="0",
    help="Number of y translations in y (for the data).",
)

parser.add_argument("--tr_prop", type=float, default="0.01", help="Train proportion.")
parser.add_argument("--te_prop", type=float, default="0.01", help="Test proportion.")
parser.add_argument("--val_prop", type=float, default="0.01", help="Valid proportion.")

parser.add_argument("--n_classes", type=int, default="300", help="Number of classes.")
parser.add_argument("--dataset", type=str, default="mnist", help="Dataset")
parser.add_argument(
    "--sftmax", type=int, default="1", help="If 1, switches to weighting and summing (deprecated softmax is always used)"
)
parser.add_argument("--tau", type=float, default=0.1, help="Temperature of softmax.")
parser.add_argument("--mode", type=str, default="train", help="training or test mode")
parser.add_argument("--supervised", type=int, default=0, help="Switches between weakly and fully supervised.")


def main(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")

    args = parser.parse_args(params)

    SEED = int(args.seed)
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if args.dataset == "simpleshapes":
        data = datasets.SimpleShapes(
            batch_size=args.bs,
            n_x_translations=args.data_n_x,
            n_y_translations=args.data_n_y,
            n_rotations=args.data_n_rot,
            n_classes=args.n_classes,
            n_pixels=28,
        )
    elif args.dataset == "mnist":
        data = datasets.ProjectiveMNIST(
            batch_size=args.bs,
            n_x_translations=args.data_n_x,
            n_y_translations=args.data_n_y,
            n_rotations=args.data_n_rot,
            train_set_proportion=args.tr_prop,
            test_set_proportion=args.te_prop,
            valid_set_proportion=args.val_prop,
        )
    if args.mode == "train":
        print("Training")
    if args.mode == "test":
        print("Testing")

    # automatically set z_dim to image size
    image_size = data.n_pixels ** 2
    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)
    dict_args = vars(args)
    save_name = "_".join(
        [
            "{0}_{1}".format(key, dict_args[key])
            for key in dict_args
            if key not in ["output_directory", "mode"]
        ]
    )

    if args.supervised:
        transformation_types = []
        indexes = []
        if args.n_rot > 0:
            transformation_types.append("ComplexShiftOperator")
            indexes.append(0)
        if args.n_x > 0:
            transformation_types.append("ComplexShiftOperator")
            indexes.append(1)
        if args.n_y > 0:
            transformation_types.append("ComplexShiftOperator")
            indexes.append(2)
        model_with_rotation = ComplexAutoEncoder(
            data,
            transformation_types=transformation_types,
            indexes=indexes,
            device=device,
            z_dim=image_size,
            seed=SEED,
            output_directory=args.output_directory,
            save_name=save_name,
            n_rotations=args.n_rot,
            n_x_translations=args.n_x,
            n_y_translations=args.n_y,
        )
        n_transfos = len(indexes)
    else:
        model_with_rotation = WeaklyComplexAutoEncoder(
            data,
            transformation_type="ComplexShiftOperator",
            device=device,
            z_dim=image_size,
            seed=SEED,
            temperature=args.tau,
            output_directory=args.output_directory,
            save_name=save_name,
            use_softmax=args.sftmax,
            n_rotations=args.n_rot,
            n_x_translations=args.n_x,
            n_y_translations=args.n_y,
        )

    if args.mode == "train":
        (
            train_loss,
            valid_loss,
            train_mse,
            valid_mse,
            test_mse,
        ) = model_with_rotation.run(n_epochs=args.n_epochs, learning_rate=args.lr)

        perf = np.array([train_mse, valid_mse, test_mse])

        torch.save(perf, os.path.join(args.output_directory, "final_mse_" + save_name))
        torch.save(
            train_loss, os.path.join(args.output_directory, "train_loss_" + save_name)
        )

        torch.save(
            valid_loss, os.path.join(args.output_directory, "valid_loss_" + save_name)
        )

        file_name = "best_checkpoint_{}.pth.tar".format(model_with_rotation.save_name)
        path_to_model = os.path.join(args.output_directory, file_name)
        best_mse, best_epoch = model_with_rotation.load_model(path_to_model)

        ##### Plots train reconstructions
        samples_pairs = np.random.randint(
            0, len(model_with_rotation.data.X_train), size=(10,)
        ).tolist()
        model_with_rotation.plot_x2_reconstructions(
            indices=samples_pairs,
            train_set=True,
            save_name=os.path.join(args.output_directory, "plots_train_reconstructions_" + save_name),
        )
        ##### Plots train rotations of samples
        train_indices = np.random.randint(
            0, len(model_with_rotation.data.X_orig_train), size=(10,)
        ).tolist()
        figsave_name=os.path.join(args.output_directory, "plots_train_rotations_" + save_name + '.png')
        if args.supervised:
            if n_transfos == 1:
                if args.data_n_x > 0:
                    param_name = 'tx'
                elif args.data_n_y > 0:
                    param_name = 'ty'
                if args.data_n_rot > 0:
                    param_name = 'angle'
                model_with_rotation.plot_multiple_transformations(indices=train_indices, train_set = True, 
                    param_name=param_name, save_name=figsave_name
                )
            else:
                model_with_rotation.plot_multiple_transformations_stacked(indices=train_indices, train_set = True, 
                    n_plots = 10, save_name=figsave_name
                )
        else:
            if args.data_n_x > 0:
                param_name = 'tx'
            elif args.data_n_y > 0:
                param_name = 'ty'
            if args.data_n_rot > 0:
                param_name = 'angle'
            model_with_rotation.plot_multiple_transformations(indices=train_indices, train_set = True, 
                param_name=param_name,save_name=figsave_name
            )

        ##### Plots test reconstructions
        samples_pairs = np.random.randint(
            0, len(model_with_rotation.data.X_test), size=(10,)
        ).tolist()
        model_with_rotation.plot_x2_reconstructions(
            indices=samples_pairs,
            train_set=False,
            save_name=os.path.join(args.output_directory, "plots_test_reconstructions_" + save_name),
        )

        ##### Plots test rotations of samples
        test_indices = np.random.randint(
            0, len(model_with_rotation.data.X_orig_test), size=(10,)
        ).tolist()
        figsave_name=os.path.join(args.output_directory, "plots_test_rotations_" + save_name + '.png')
        if args.supervised:
            if n_transfos == 1:
                if args.data_n_x > 0:
                    param_name = 'tx'
                elif args.data_n_y > 0:
                    param_name = 'ty'
                if args.data_n_rot > 0:
                    param_name = 'angle'
                model_with_rotation.plot_multiple_transformations(indices=test_indices, train_set = False, 
                    param_name=param_name, save_name=figsave_name
                )
            else:
                model_with_rotation.plot_multiple_transformations_stacked(indices=test_indices, train_set = False, 
                    n_plots = 10, save_name=figsave_name
                )
        else:
            if args.data_n_x > 0:
                param_name = 'tx'
            elif args.data_n_y > 0:
                param_name = 'ty'
            if args.data_n_rot > 0:
                param_name = 'angle'
            model_with_rotation.plot_multiple_transformations(indices=test_indices, train_set = False, 
                param_name=param_name, save_name=figsave_name
            )

        
    elif args.mode == "test":
        file_name = "best_checkpoint_{}.pth.tar".format(model_with_rotation.save_name)
        path_to_model = os.path.join(args.output_directory, file_name)
        model_with_rotation.load_model(path_to_model)
        if args.supervised:
            loss_func = model_with_rotation.reconstruction_mse_transformed_z1
        else:
            loss_func = model_with_rotation.reconstruction_mse_transformed_z1_weak
        test_mse = model_with_rotation.compute_test_loss(
            loss_func, model_with_rotation.data.test_loader_batch_100
        )
        torch.save(
            torch.FloatTensor([test_mse]),
            os.path.join(
                args.output_directory, "test_mse_" + model_with_rotation.save_name
            ),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
