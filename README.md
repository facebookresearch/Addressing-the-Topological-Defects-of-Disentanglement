# Addressing the Topological Defects of Disentanglement via Distributed Operators

[ArXiv link](https://arxiv.org/abs/2102.05623)

## Installation

To install the required packages:
`pip install -r requirements.txt`

## Structure

The repo contains all code to reproduce experiments along with the code for generating the datasets: 

* Datasets and transformations can be found in `datasets/`
* Autoencoder, Shift Operator, and Disentangled Operator models are implemented in `autoencoder.py` and `latent_operators.py`
* CCI VAE baselines and variants (simple VAE and Beta-VAE) are in `cci_variational_autoencoder.py`
* Weakly supervised versions of the shift operator are in `weakly_complex_shift_autoencoder.py`
* Complex and stacked shift operator models with multiple transformation layers are in `complex_shift_autoencoder.py`
* All model architectures are defined in `models.py`

## Model Training


### VAE Baselines (simple VAE, Beta-VAE and CCI-VAE)
You can train any VAE baseline model via:
`python run_experiments_real.py [name] --model cci_vae [arguments]`

For example, to run CCI VAE and variants on rotated digits:
`python run_experiments_real.py cci-vae-single-digit-mnist-rotations --model cci_vae --data single_digit_mnist --mnist_proportion 1.0`

For a full list of experiments and options available, see below:
```
positional arguments:
  name                  name of experiment

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         model for experiments. Example: autoencoder, cci_vae
  --data DATA           dataset used for training: mnist, single_digit_mnist
  --mnist_proportion MNIST_PROPORTION
                        proportion of mnist to use
  --n_classes N_CLASSES
                        number of classes to use for simple shapes
  --z_dim Z_DIM         dataset used for training
  --transformation {rotation,shift_x,shift_y}
  --distribution {gaussian,bernoulli}
                        likelihood distribution used for computing loss in CCI
                        VAE
  --beta BETA           beta used for CCI VAE and Beta-VAE
```


### Autoencoder, Shift Operator, and Disentangled Operator
To train a standard autoencoder or one with latent operators:
`python run_experiments_real.py [name] --model autoencoder [arguments]`

For example, to train a standard autoencoder with 30 latents:
`python run_experiments_real.py standard-autoencoder-shapes --model autoencoder --z_dim 3k --data shapes --no_latent_op`

To train the Shift and Disentangled operator models on shapes with 30 latents: 
`python run_experiments_real.py shift-operator-on-shapes --model autoencoder --z_dim 30 --data shapes --transformation rotations`

For a full list of experiments and options available, see below:
```
positional arguments:
  name                  name of experiment

optional arguments:
  -h, --help            show this help message and exit
  --architecture ARCHITECTURE
                        name of autoencoder architecture: CCI or Linear.
  --data DATA           dataset used for training: mnist, single_digit_mnist
  --mnist_proportion MNIST_PROPORTION
                        proportion of mnist to use
  --n_classes N_CLASSES
                        number of classes to use for simple shapes
  --z_dim Z_DIM         dataset used for training
  --transformation {rotation,shift_x,shift_y}
  --distribution {gaussian,bernoulli}
                        likelihood distribution used for computing loss in CCI
                        VAE
  --no_latent_op        use standard autoencoder without latent operators
```

### Complex Shift Operator and Weakly Supervised

* To train the Stacked shift operator on shapes with 4 rotations, 5 translations in x and 5 translations in y: 

`cd complex_shift_operator`

`python __main__.py --lr 0.0005 --n_rot 3 --data_n_rot 3 --n_x 4 --data_n_x 4 --n_y 4 --data_n_y 4 --supervised 1 --dataset simpleshapes --bs 32 --n_epochs 5`

This model is slow to train due to the high number of pairs generated (as rotations and translations are used jointly), you can use `--n_classes 30` for quicker results.
Note that importantly n_rot and data_n_rot (respectively n_x and data_n_x, and n_y and data_n_y) should have the same values, as this is the supervised shift complex operator.

* To train the weakly supervised shift operator on shapes with 10 rotations (the weakly supervised operator only handles 1 type of transformations at a time): 

`cd complex_shift_operator`

`python __main__.py --lr 0.0005 --n_rot 9 --data_n_rot 9 --n_x 0 --data_n_x 0 --n_y 0 --data_n_y 0 --tau 0.1 --supervised 0 --dataset simpleshapes --bs 32 --n_epochs 5`

Note that this model, the number of transformations of the data and in the model can be different as it is the unsupervised version. If the number of transformation in the model (referred to as K_L in the paper) is smaller than in the data, plotting function will throw an error. 

Both commands will create a directory (specified by --output_directory, default name is output) where the model for the epoch with lowest validation loss is saved (under `best_checkpoint_` + name including hyper-parameters).

For a full list arguments available, see below:

```
  -h, --help            show this help message and exit
  --seed SEED
  --output_directory OUTPUT_DIRECTORY
                        In this directory the models will be saved. Will be
                        created if doesn't exist.
  --n_epochs N_EPOCHS   Number of epochs.
  --lr LR               Learning rate.
  --bs BS               Batch size.
  --n_rot N_ROT         Number of rotations (for the model).
  --data_n_rot DATA_N_ROT
                        Number of rotations (for the data).
  --n_x N_X             Number of x translations in x (for the model).
  --data_n_x DATA_N_X   Number of x translations in x (for the data).
  --n_y N_Y             Number of y translations in y (for the model).
  --data_n_y DATA_N_Y   Number of y translations in y (for the data).
  --tr_prop TR_PROP     Train proportion.
  --te_prop TE_PROP     Test proportion.
  --val_prop VAL_PROP   Valid proportion.
  --n_classes N_CLASSES
                        Number of classes.
  --dataset DATASET     Dataset
  --sftmax SFTMAX       If 1, switches to weighting and summing (deprecated
                        softmax is always used)
  --tau TAU             Temperature of softmax.
  --mode MODE           training or test mode
  --supervised SUPERVISED
                        Switches between weakly and fully supervised.

```

* The number of translations or rotations specified via `data_n_rot, n_x, n_y` excludes the identity. For example, `data_n_rot = 9` yields a total of 10 rotations (9 rotations plus the identity).

Note running experiments with a very small number of classes for simple shapes may trigger a warning if not enough samples are present for validation. If so, please increase the number of classes and rerun.

## Tests

To run unittests: `python -m pytest tests`

## License

See [License file](LICENSE)
