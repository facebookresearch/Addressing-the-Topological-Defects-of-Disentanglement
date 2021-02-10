"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Launches experiments locally or on the cluster

python run_experiments.py [name] --cluster

OPTIONS:
python run_experiments.py linear-mnist-test --data mnist
python run_experiments.py cci-autoencoder-shapes --architecture CCI
"""
import argparse
import autoencoder
import cci_variational_autoencoder
import os
import itertools
from datasets import datasets
from functools import partial
import torch
import shutil
import submitit


BASE_PARAMS = {
    "seed": [0, 10, 20, 30, 40],
    "n_epochs": [30],
    "learning_rate": [0.001, 0.0005],
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"running on {device}")


def run_cci_vae_shapes(
    beta=1000.0,
    c_max=36.0,
    z_dim=30,
    batch_size=16,
    n_epochs=10,
    learning_rate=0.0005,
    seed=0,
    folder=None,
    n_classes=300,
    architecture=None,
    n_rotations=9,
    n_x_translations=0,
    n_y_translations=0,
    distribution="gaussian",
):
    """Runs CCI VAE and variants on Simple Shapes. Note architecture kwarg is not used"""
    if folder is None:
        raise ValueError("Please provide an experiment folder")
    print("saving results to ", folder)

    shapes = datasets.SimpleShapes(
        batch_size,
        n_rotations=n_rotations,
        n_x_translations=n_x_translations,
        n_y_translations=n_y_translations,
        n_classes=n_classes,
        seed=seed,
        pairs=False,
    )
    train_cci_vae_variants(
        shapes, beta, c_max, z_dim, n_epochs, learning_rate, distribution, seed, folder
    )


def run_cci_vae_mnist(
    beta=1000.0,
    c_max=36.0,
    z_dim=30,
    batch_size=16,
    n_epochs=10,
    learning_rate=0.0005,
    seed=0,
    folder=None,
    n_classes=300,
    proportion=0.01,
    architecture=None,
    n_rotations=9,
    n_x_translations=0,
    n_y_translations=0,
    distribution="gaussian",
):
    """Runs CCI VAE and variants on MNIST. Note architecture kwarg is not used"""
    if folder is None:
        raise ValueError("Please provide an experiment folder")
    print("saving results to ", folder)

    mnist = datasets.ProjectiveMNIST(
        batch_size,
        seed=seed,
        train_set_proportion=proportion,
        test_set_proportion=1.0,
        valid_set_proportion=proportion,
        n_rotations=n_rotations,
        n_x_translations=n_x_translations,
        n_y_translations=n_y_translations,
        pairs=False,
    )
    train_cci_vae_variants(
        mnist, beta, c_max, z_dim, n_epochs, learning_rate, distribution, seed, folder
    )


def run_cci_vae_single_digit_mnist(
    beta=1000.0,
    c_max=36.0,
    z_dim=30,
    batch_size=16,
    n_epochs=10,
    learning_rate=0.0005,
    seed=0,
    folder=None,
    n_classes=300,
    proportion=0.01,
    architecture=None,
    n_rotations=9,
    n_x_translations=0,
    n_y_translations=0,
    distribution="gaussian",
):
    """Runs CCI VAE and variants on MNIST. Note architecture kwarg is not used"""
    if folder is None:
        raise ValueError("Please provide an experiment folder")
    print("saving results to ", folder)

    mnist = datasets.ProjectiveSingleDigitMNIST(
        batch_size,
        seed=seed,
        train_set_proportion=proportion,
        test_set_proportion=1.0,
        valid_set_proportion=proportion,
        n_rotations=n_rotations,
        n_x_translations=n_x_translations,
        n_y_translations=n_y_translations,
        pairs=False,
    )
    train_cci_vae_variants(
        mnist, beta, c_max, z_dim, n_epochs, learning_rate, distribution, seed, folder
    )


def train_cci_vae_variants(
    data, beta, c_max, z_dim, n_epochs, learning_rate, distribution, seed, folder
):
    """Trains CCI, Beta, and standard VAE"""
    print("Training CCI VAE")
    cci_vae_folder = os.path.join(folder, "cci_vae")
    train_cci_vae(
        data,
        beta,
        c_max,
        z_dim,
        n_epochs,
        learning_rate,
        distribution,
        seed,
        cci_vae_folder,
    )
    print("Training Beta VAE")
    beta_vae_folder = os.path.join(folder, "beta_vae")
    train_cci_vae(
        data,
        beta,
        0.0,
        z_dim,
        n_epochs,
        learning_rate,
        distribution,
        seed,
        beta_vae_folder,
    )
    print("Training VAE")
    vae_folder = os.path.join(folder, "vae")
    train_cci_vae(
        data, 1.0, 0.0, z_dim, n_epochs, learning_rate, distribution, seed, vae_folder
    )


def run_autoencoder_shapes(
    z_dim=1000,
    batch_size=16,
    n_epochs=30,
    learning_rate=0.0005,
    seed=0,
    folder=None,
    architecture="Linear",
    n_classes=300,
    n_rotations=9,
    n_x_translations=0,
    n_y_translations=0,
    distribution=None,
    use_latent_op=True,
):
    if folder is None:
        raise ValueError("Please provide an experiment folder")
    print("saving results to ", folder)

    shapes = datasets.SimpleShapes(
        batch_size,
        n_classes=n_classes,
        seed=seed,
        n_rotations=n_rotations,
        n_x_translations=n_x_translations,
        n_y_translations=n_y_translations,
    )
    if use_latent_op:
        train_autoencoder(
            shapes, z_dim, n_epochs, learning_rate, seed, folder, architecture
        )
    else:
        train_standard_autoencoder(
            shapes, z_dim, n_epochs, learning_rate, seed, folder, architecture
        )


def run_autoencoder_mnist(
    z_dim=1000,
    batch_size=16,
    n_epochs=2,
    learning_rate=0.0005,
    seed=0,
    folder=None,
    architecture="Linear",
    proportion=0.01,
    n_rotations=9,
    n_x_translations=0,
    n_y_translations=0,
    distribution=None,
    use_latent_op=True,
):
    if folder is None:
        raise ValueError("Please provide an experiment folder")
    print("saving results to ", folder)

    mnist = datasets.ProjectiveMNIST(
        batch_size,
        seed=seed,
        train_set_proportion=proportion,
        test_set_proportion=1.0,
        valid_set_proportion=proportion,
        n_rotations=n_rotations,
        n_x_translations=n_x_translations,
        n_y_translations=n_y_translations,
    )
    if use_latent_op:
        print("using latent_op")
        train_autoencoder(
            mnist, z_dim, n_epochs, learning_rate, seed, folder, architecture
        )
    else:
        train_standard_autoencoder(
            mnist, z_dim, n_epochs, learning_rate, seed, folder, architecture
        )


def train_standard_autoencoder(
    data, z_dim, n_epochs, learning_rate, seed, folder, architecture
):
    model = autoencoder.AutoEncoder(
        data,
        z_dim=z_dim,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        encoder_type=architecture,
        decoder_type=architecture,
        device=device,
        seed=seed,
    )
    model.run()
    model.save_best_validation(os.path.join(folder, "standard-autoencoder"))


def train_autoencoder(data, z_dim, n_epochs, learning_rate, seed, folder, architecture):

    model_disentangled_rotation = autoencoder.AutoEncoder(
        data,
        z_dim=z_dim,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        latent_operator_name="DisentangledRotation",
        encoder_type=architecture,
        decoder_type=architecture,
        device=device,
        seed=seed,
    )
    model_disentangled_rotation.run()
    model_disentangled_rotation.save_best_validation(
        os.path.join(folder, "disentangled-operator")
    )

    model_shift_operator = autoencoder.AutoEncoder(
        data,
        z_dim=z_dim,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        latent_operator_name="ShiftOperator",
        encoder_type=architecture,
        decoder_type=architecture,
        device=device,
        seed=seed,
    )
    model_shift_operator.run()
    model_shift_operator.save_best_validation(os.path.join(folder, "shift-operator"))


def train_cci_vae(
    data, beta, c_max, z_dim, n_epochs, learning_rate, distribution, seed, folder
):
    cci_vae = cci_variational_autoencoder.CCIVariationalAutoEncoder(
        data,
        beta=beta,
        c_max=c_max,
        z_dim=z_dim,
        seed=seed,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        distribution=distribution,
    )
    cci_vae.train()
    cci_vae.save_best_validation(folder)


def launch_single_job(experiment, base_dir, results_dir, **kwargs):
    log_folder = base_dir + "%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    # executor.update_parameters(timeout_min=600, gpus_per_node=1)
    executor.update_parameters(
        timeout_min=240, gpus_per_node=1,
    )
    job = executor.submit(experiment, folder=results_dir, **kwargs)
    print("job id", job.job_id)
    print(f"logging to: {base_dir + job.job_id}")
    print(f"results stored at: {results_dir}")

    result = job.result()
    print(f"job result: {result}")


def launch_sweep(experiment, params, base_dir, experiment_dir):
    log_folder = base_dir + "%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    # executor.update_parameters(timeout_min=600, gpus_per_node=1)
    executor.update_parameters(
        timeout_min=600, gpus_per_node=1,
    )
    jobs = []
    with executor.batch():
        for i, param in enumerate(params):
            print("running with param ", param)
            param["folder"] = os.path.join(experiment_dir, f"{i}")
            job = executor.submit(experiment, **param)
            jobs.append(job)
    print(f"launched {len(params)} jobs")
    print("sweep id", jobs[0].job_id)
    print(f"logging to: {base_dir}{jobs[0].job_id}")

    results = [job.result() for job in jobs]
    print(f"job results: {results}")


def get_params(args):
    params = BASE_PARAMS
    if args.data == "mnist":
        params["batch_size"] = [8, 16, 32, 64]
    elif args.data == "shapes":
        params["batch_size"] = [4, 8, 16, 32]

    if args.model == "cci_vae":
        params["n_epochs"] = [10, 20, 50]
        params["beta"] = [4.0, 10.0, 100.0, 1000.0]
        params["z_dim"] = [10, 30]
    return params


def get_param_combinations(params):
    """Returns a list of dictionaries with all combinations"""
    keys, values = zip(*params.items())
    params_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return params_combinations


def get_directories(args, cluster=False):
    user = os.environ["USER"]
    if cluster:
        RESULTS_DIR = f"/checkpoint/{user}/Equivariance/"
        base_dir = f"/checkpoint/{user}/jobs/{args.name}/"
    else:
        RESULTS_DIR = os.path.expanduser(
            "~/Dropbox/FAIR/Projects/Equivariance/experiments/results"
        )
        base_dir = os.path.expanduser(
            "~/Dropbox/FAIR/Projects/Equivariance/experiments/jobs/{args.name}/"
        )
    experiment_dir = os.path.join(RESULTS_DIR, args.name)
    # clean experimental directory
    if os.path.exists(experiment_dir):
        shutil.rmtree(experiment_dir)
    return base_dir, experiment_dir


def get_experiment_function(args):
    experiments = {
        "run_autoencoder_shapes": run_autoencoder_shapes,
        "run_autoencoder_mnist": run_autoencoder_mnist,
        "run_cci_vae_shapes": run_cci_vae_shapes,
        "run_cci_vae_mnist": run_cci_vae_mnist,
        "run_cci_vae_single_digit_mnist": run_cci_vae_mnist,
    }
    experiment = experiments[f"run_{args.model}_{args.data}"]
    print(f"run_{args.model}_{args.data}")

    if args.data == "shapes":
        experiment = partial(experiment, n_classes=args.n_classes)
    elif args.data in {"mnist", "single_digit_mnist"}:
        experiment = partial(experiment, proportion=args.mnist_proportion)
    else:
        raise ValueError(f"dataset {args.data} not supported")

    # standard autoencoder
    if "autoencoder" == args.model and args.no_latent_op:
        experiment = partial(experiment, use_latent_op=False)

    n_rotations, n_x_translations, n_y_translations = get_n_transformations(args)

    experiment = partial(
        experiment,
        n_rotations=n_rotations,
        n_x_translations=n_x_translations,
        n_y_translations=n_y_translations,
        architecture=args.architecture,
        z_dim=args.z_dim,
        distribution=args.distribution,
    )
    return experiment


def get_n_transformations(args):
    n_rotations, n_x_translations, n_y_translations = 0, 0, 0
    n_transformations = 9

    if args.transformation == "rotation":
        n_rotations = n_transformations
    if args.transformation == "shift_x":
        n_x_translations = n_transformations
    if args.transformation == "shift_y":
        n_y_translations = n_transformations
    return n_rotations, n_x_translations, n_y_translations


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="python run_experiments --cluster",
        description="runs experiments with specified parameters",
    )
    parser.add_argument("name", help="name of experiment")
    parser.add_argument(
        "--model",
        help="model for experiments. Example: autoencoder, cci_vae",
        default="autoencoder",
    )
    parser.add_argument(
        "--architecture", help="name of autoencoder architecture", default="Linear",
    )
    parser.add_argument(
        "--data",
        help="dataset used for training: mnist, single_digit_mnist",
        default="shapes",
    )
    parser.add_argument(
        "--mnist_proportion",
        help="proportion of mnist to use",
        default=0.01,
        type=float,
    )
    parser.add_argument(
        "--n_classes",
        help="number of classes to use for simple shapes",
        default=300,
        type=int,
    )
    parser.add_argument(
        "--z_dim", help="dataset used for training", default=1000, type=int
    )
    parser.add_argument(
        "--transformation",
        choices=["rotation", "shift_x", "shift_y"],
        type=str.lower,
        default="rotation",
    )
    parser.add_argument(
        "--distribution",
        help="likelihood distribution used for computing loss in CCI VAE",
        choices=["gaussian", "bernoulli"],
        type=str.lower,
        default="gaussian",
    )
    parser.add_argument("--beta", help="beta used for CCI VAE", default=1000, type=int)
    parser.add_argument(
        "--no_latent_op",
        help="use standard autoencoder without latent operators",
        action="store_true",
    )
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    return parser


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()

    experiment = get_experiment_function(args)

    base_dir, experiment_dir = get_directories(args, cluster=args.cluster)

    if args.cluster and args.sweep:
        params = get_params(args)
        params_combinations = get_param_combinations(params)
        launch_sweep(experiment, params_combinations, base_dir, experiment_dir)
    elif args.cluster:
        launch_single_job(
            experiment, base_dir, experiment_dir,
        )
    else:
        print("running single local job")
        experiment(folder=experiment_dir)
