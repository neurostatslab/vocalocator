# Vocal Call Locator

## Installation

### Pip:
1. (Optional) Create a virtual environment: `python -m venv vcl_env && source vcl_env/bin/activate`
2. `pip install vocalocator`

### Conda:
1. Install Conda for your system with the [instructions here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
2. Clone this repository: `git clone https://github.com/neurostatslab/vocalocator.git && cd vocalocator`
3. Create an environment and install: `conda create -n vcl -f environment_conda.yml`

### Pipenv:
1. Clone this repository: `git clone https://github.com/neurostatslab/vocalocator.git && cd vocalocator`
2. Install Pipenv: `pip install pipenv`
3. Install package: `pipenv install`
4. Enter the pipenv shell to access the new virtual environment: `pipenv shell`


## Quick Start
TODO

## Advanced Usage
1. Create a dataset. This should be an HDF5 file with the following datasets:

| Dataset group/name | Shape             | Data type | Description                                                                                                                                    |
|--------------------|-------------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------|
| /audio     | (*, n_channels) | float     | All sound events concatenated along axis 0                                                                                                     |
| /length_idx           | (n + 1,)          | int       | Index into audio dataset. Sound event `i` should span the half open interval [`length_idx[i]`, `length_idx[i+1]`) and the first element should be 0. |
| /locations         | (n, 2)            | float     | Locations associated with each sound event. Only required for training.                                                                        |
1. Optionally, manually generate a train/val split. If none is provided, an 80/10/10 train/validation/test split will be generated automatically and saved in the model directory. Manually generated splits can be used by saving them to the same directory as 1-D numpy arrays with the names `train_set.npy` and `val_set.npy`. This directory is passed to VCL through the `--indices` option.
```
# Simple script to generate a test-train split
import numpy as np
dataset_size = <INSERT DATASET SIZE>
train_size, val_size = int(0.8 * dataset_size), int(0.1 * dataset_size)
train_set, val_set, test_set = np.split(np.random.permutation(dataset_size), [train_size, train_size + val_size])
np.save('train_set.npy', train_set)
np.save('val_set.npy', val_set)
np.save('test_set.npy', test_set)
```
2. Create a config. This is a JSON file consisting of a single object whose properties correspond to the hyperparameters of the model and optimization algorithm. See examples in the sample_configs directory
3. Train a model: `python -m vocalocator --data /path/to/directory/containing/trainset/ --config /path/to/config.json --save-path /path/to/model/weight/directory/ --indices /optional/path/to/index/directory`
   *  _(Optional)_ Initialize with pretrained weights by populating the "WEIGHTS_PATH" field in the top-level of the config file with a path to the saved model state as a .pt file.
5. Using the trained model, perform inference: `python -m vocalocator.assess --inference --data /path/to/hdf5/dataset.h5 --config /path/to/model_dir/config.json -o /optional/output/path.h5 --index /optional/index/path.npy`.
   * Note that here, the config should point toward a config.json in a trained model's directory. This ensures the "WEIGHTS_PATH" field exists and contains the path to the weights corresponding to the best-performing epoch of the model.  
   * Output predictions will be stored in a dataset labeled `point_predictions` at the root of the HDF5 file.

## Public datasets
See our [dataset website](https://users.flatironinstitute.org/~atanelus/) to learn more about and download our public datasets.