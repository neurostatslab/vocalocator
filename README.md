gerbilizer
==============================

Gerbil audio localization

Project Organization
------------

    ├── LICENSE
    ├── README.md                               <- The top-level README for developers using this project.
    ├── data                                    <- Data
    │
    ├── trained_models                          <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                              and a short `-` delimited description, e.g.
    │                                              `1.0-initial-data-exploration`.
    │
    ├── requirements.txt                        <- The requirements file for reproducing the analysis environment
    │
    ├── training                                <- Scripts to create new models and train them.


Thoughts and Notes
-----------------

AHW (01/24/22)
* I used a [`DenseNet`](https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_ResNet_or_DenseNet_Introducing_Dense_Shortcuts_to_ResNet_WACV_2021_paper.pdf) architecture to start. At each layer, the number of timebins is halved because I use a stride length of two. The previous layer activations are concatenated onto the output after downsampling by a factor of two (using average pooling). The idea here is that we can extract features at coarser time resolutions the deeper we go into the network.
* I used the activation function (gated tanh) used in [WaveNet](https://arxiv.org/abs/1609.03499), which is one of the more impressive speech generation networks I've seen. They use ResNet layers, but I used DenseNet layers (see link above).
* Should we pre-compute all pairwise cross-correlations of the four microphones and feed this into the network at layer 1? I think this might help, since we know this is a good feature that human-engineered solutions for audio localization use.
* I made a script `data/reshape_dataset.py` that lightly pre-processes the data. One thing we should think about is what units the audio inputs and spatial outputs should be. Right now I am multiplying the audio waveforms by `1e3` and the locations by `1e-3`, as this seemed to produce network that varied from input to input at initialization. We should play around with this more systematically though...
* The `notebooks/0001-visualize-predictions.ipynb` file can be used to visualize network predictions and compare them to the ground truth. You can load `init_weights` or `best_weights` to view predictions of the initial model (before training) and the best model found during training.
* We should prioritize data cleaning... My hope is that by getting rid of bad samples we'll get a big bump in performance. We should filter out the easy ones (frames where nose is super far away from ears), and then try spot-checking a few hundred random frames




