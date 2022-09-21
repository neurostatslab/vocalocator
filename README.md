# Gerbilizer
==============================

## Gerbil audio localization

## Project Organization
------------

    ├── LICENSE
    ├── README.md                               <- The top-level README for developers using this project.
    │
    ├── trained_models                          <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                              and a short `-` delimited description, e.g.
    │                                              `1.0-initial-data-exploration`.
    │
    ├── requirements.txt                        <- The requirements file for reproducing the analysis environment
    │
    ├── gerbilizer                                <- Scripts to create new models and train them.


## Installation
1. Clone this repository: `git clone https://github.com/neurostatslab/gerbilizer.git`
2. Install prerequisites: `cd gerbilizer & pipenv install`
3. Install with pip: `pip install .`

## Usage
1. Create a dataset. This should be an HDF5 file with the folliwng datasets:

| Dataset group/name | Shape             | Data type | Description                                                                                                                                    |
|--------------------|-------------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------|
| /vocalizations     | (*, n_channels) | float     | All sound events concatenated along axis 0                                                                                                     |
| /len_idx           | (n + 1,)          | int       | Index into vocalizations. Sound event `i` should span the half open interval [`len_idx[i]`, `len_idx[i+1]`) and the first element should be 0. |
| /locations         | (n, 2)            | float     | Locations associated with each sound event. Only required for training.                                                                        |
1. Optionally, partition the dataset into a test set, validation set, and training set. Otherwise, place the dataset in an appropriate directory and rename it to **train_set.h5**
2. Create a config. This is a JSON file consisting of a single object whose properties correspond to the hyperparameters of the model and optimization algorithm.
3. Train a model:
   1. With SLURM: `sbatch run_gpu.sh /path/to/directory/containing/trainset/ /path/to/config.json`. Note that the first argument is expected to be a directory and the second argument is expected to be a file
   2. Without SLURM: `python -m gerbilizer.main --data /path/to/directory/containing/trainset/ --config /path/to/config.json --save_path /path/to/model/weight/directory/`
4. Optionally, run with pretrained weights by pointing to the config.json within the model weight directory of the pretrained model.
5. Perform inference:
   1. With SLURM: `sbatch run_eval.sh /path/to/hdf5/dataset.h5 /path/to/model_dir/trained_models/config_name/#####/config.json /optional/output/path.h5`. 
   2. Without SLURM: `python -m gerbilizer.main --eval --data /path/to/hdf5/dataset.h5 --config /path/to/model_dir/trained_models/config_name/#####/config.json -o /optional/output/path.h5`.
   3. Note that here, the data argument is a file rather than a directory and the config should point toward a config.json in a trained model's directory, just as in the finetuning step. If no output path is provided, predictions will be stored in the same directory as the input dataset.
   4. Output predictions will be stored in a dataset labeled "predictions" at the root of the HDF5 hierarchy


## Thoughts and Notes
-----------------

AHW (01/24/22)
* I used a [`DenseNet`](https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_ResNet_or_DenseNet_Introducing_Dense_Shortcuts_to_ResNet_WACV_2021_paper.pdf) architecture to start. At each layer, the number of timebins is halved because I use a stride length of two. The previous layer activations are concatenated onto the output after downsampling by a factor of two (using average pooling). The idea here is that we can extract features at coarser time resolutions the deeper we go into the network.
* I used the activation function (gated tanh) used in [WaveNet](https://arxiv.org/abs/1609.03499), which is one of the more impressive speech generation networks I've seen. They use ResNet layers, but I used DenseNet layers (see link above).
* Should we pre-compute all pairwise cross-correlations of the four microphones and feed this into the network at layer 1? I think this might help, since we know this is a good feature that human-engineered solutions for audio localization use.
* I made a script `data/reshape_dataset.py` that lightly pre-processes the data. One thing we should think about is what units the audio inputs and spatial outputs should be. Right now I am multiplying the audio waveforms by `1e3` and the locations by `1e-3`, as this seemed to produce network that varied from input to input at initialization. We should play around with this more systematically though...
* The `notebooks/0001-visualize-predictions.ipynb` file can be used to visualize network predictions and compare them to the ground truth. You can load `init_weights` or `best_weights` to view predictions of the initial model (before training) and the best model found during training.
* We should prioritize data cleaning... My hope is that by getting rid of bad samples we'll get a big bump in performance. We should filter out the easy ones (frames where nose is super far away from ears), and then try spot-checking a few hundred random frames
* Feed in the Hilbert transformed signals?

