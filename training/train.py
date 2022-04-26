import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable, NewType, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

from configs import build_config_from_file, build_config_from_name
from models import build_model
from dataloaders import build_dataloaders
from augmentations import build_augmentations
from logger import ProgressLogger


logger = logging.getLogger('no_spam')
logger.setLevel(logging.DEBUG)


JSON = NewType('JSON', dict)
base_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# =============================== #
# === Parse Command Line Args === #
# =============================== #

def get_args():
    parser = argparse.ArgumentParser()

    # Configs can be provided as either a name, which then references an entry in the dictionary
    # located in configs.py, or as a path to a JSON file, when then uses the entries in that file
    # to override the default configuration entries.

    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Used to specify model configuration via a hard-coded named configuration.",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=False,
        help="Used to specify model configuration via a JSON file."
    )

    parser.add_argument(
        "--job_id",
        type=int,
        required=False,
        help="Used to seed random hyperparameters and save output.",
    )

    parser.add_argument(
        "--datafile",
        type=str,
        required=False,
        default=os.path.join(base_dir, "data"),
        help="Path to vocalization data.",
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args):
    if not os.path.exists(args.datafile):
        raise ValueError(f"Error: could not find data at path {args.datafile}")

    if args.config_file is not None and not os.path.exists(args.config_file):
        raise ValueError(f"Requested config JSON file could not be found: {args.config_file}")
    
    if args.config_file is None and args.config is None:
        args.config = "default"

    # Although it's somewhat inappropriate, I've elected to load config JSON here because a
    # thorough validation of the user input involves validating the presently unloaded JSON
    if args.config_file is not None:
        args.config_data = build_config_from_file(args.config_file, args.job_id)
    else:
        # Only runs when config_file is not provided
        # Although the intention is for the user to only provide one of the two, this
        # prioritizes config_file for the sake of eliminating undefined behavior
        args.config_data = build_config_from_name(args.config, args.job_id)
    
    if args.job_id is None:
        job_dir = os.path.join(
            base_dir,
            "trained_models",
            args.config_data['CONFIG_NAME']
        )
        if not os.path.exists(job_dir):
            args.job_id = 1
        else:
            dirs = [f for f in os.listdir(job_dir) if os.path.isdir(os.path.join(job_dir, f))]
            # Finds the highest job id and adds 1, defualts to 1 if there are no existing jobs
            args.job_id = 1 if not dirs else 1 + max(int(d) for d in dirs)
        args.config_data['JOB_ID'] = args.job_id


class Trainer():
    def __init__(self,
        datafile: str,
        config_data: JSON,
        job_id: int, *,
        eval_mode: bool=False
    ):
        self._eval = eval_mode
        self._datafile = datafile
        self._config = config_data
        self._job_id = job_id
        if not self._eval:
            self._setup_dataloaders()
            self._setup_output_dirs()
            self._augment = build_augmentations(self._config)

        # setup_model depends on logger setup in setup_output_dirA
        # thus, the if statement on _eval is split in two
        self._setup_model()

        if not self._eval:
            logger.info(f" ==== STARTING TRAINING ====\n")
            logger.info(f">> SAVING INITIAL MODEL WEIGHTS TO {self.init_weights_file}")
            self.save_weights(self.init_weights_file)

            self._best_loss = float('inf')
            self.last_completed_epoch = 0
            self.loss_history = list()

    def train_epoch(self, epoch_num):
        # Set the learning rate using cosine annealing.
        new_lr = self._get_lr(epoch_num)
        logger.info(f">> SETTING NEW LEARNING RATE: {new_lr}")
        for param_group in self.optim.param_groups:
            param_group['lr'] = new_lr

        self.progress_log.start_training()
        self.model.train()
        for sounds, locations in self.traindata:
            # Don't process partial batches.
            if len(sounds) != self._config["TRAIN_BATCH_SIZE"]:
                break

            # Move data to gpu, if desired.
            if self._config["DEVICE"] == "GPU":
                sounds = sounds.cuda()
                locations = locations.cuda()

            # Prepare optimizer.
            self.optim.zero_grad()

            # Forward pass, including data augmentation.
            outputs = self.model(
                self._augment(sounds, sample_rate=16000)
            )

            # Compute loss.
            losses = self._loss_fn(outputs, locations)
            mean_loss = torch.mean(losses)

            # Backwards pass.
            mean_loss.backward()
            self.optim.step()

            # Count batch as completed.
            self.progress_log.log_train_batch(
                mean_loss.item(), np.nan, len(sounds)
            )

            if self._config['SAVE_LOSS_PLOT']:
                self.loss_history.append(mean_loss.detach().cpu().item())
            
        self.last_completed_epoch = epoch_num + 1

    def eval_validation(self):
        self.progress_log.start_testing()
        self.model.eval()
        with torch.no_grad():
            for sounds, locations in self.valdata:
                # Move data to gpu
                if self._config["DEVICE"] == "GPU":
                    sounds = sounds.cuda()
                    locations = locations.cuda()

                # Forward pass.
                outputs = self.model(sounds)

                # Compute loss.
                losses = self._loss_fn(outputs, locations)
                mean_loss = torch.mean(losses)

                # Log progress
                self.progress_log.log_val_batch(
                    mean_loss.item(), np.nan, len(sounds)
                )
        
        if self._config['SAVE_SAMPLE_OUTPUT']:
            maps_np = outputs.detach().cpu().numpy()
            labels_np = locations.detach().cpu().numpy()
            np.save(
                os.path.join(self.sample_output_dir, 'epoch_{:0>4d}_pred.npy'.format(self.last_completed_epoch)),
                maps_np
            )
            np.save(
                os.path.join(self.sample_output_dir, 'epoch_{:0>4d}_true.npy'.format(self.last_completed_epoch)),
                labels_np
            )
        
        if self._config['SAVE_LOSS_PLOT']:
            try:
                plt.plot(self.loss_history, 'r-')
                plt.xlabel('Minibatch #')
                plt.ylabel('Loss (arb. unit)')
                plt.title("Training loss evolution")
                plt.yscale('log')
                plt_path = os.path.join(self.output_dir, 'loss_fig.png')
                plt.savefig(plt_path)
            except:
                pass  # If the loss is negative, things can go wrong here

        # Done with epoch.
        val_loss = self.progress_log.finish_epoch()

        # Save best set of weights.
        if val_loss < self._best_loss:
            self._best_loss = val_loss
            logger.info(
                f">> VALIDATION LOSS IS BEST SO FAR, SAVING WEIGHTS TO {self.best_weights_file}"
            )
            self.save_weights(self.best_weights_file)
        logger.info(self._query_mem_usage())
    
    def eval_on_dataset(self, dset, batch_size=None):
        """ Creates a generator that sequentially evaluates the model on the data within
        some dataset. The dataset is expected to be an h5py File object with a dataset labeled
        'vocalizations' within the root group. This dataset should have shape 
        (n, n_channels, n_samples), with n_channels corresponding to the channel count the model
        was trained to handle.
        """
        self.model.eval()
        if batch_size is None:
            batch_size = self._config['TRAIN_BATCH_SIZE']
        use_gpu = self._config['DEVICE'] == 'GPU'
        pad_len = 8192  # I think I should be able to store this in the model parameter dict

        vocalizations = dset['vocalizations']
        n_vocs = vocalizations.shape[0]
        idx = 0
        with torch.no_grad():
            while idx < n_vocs:
                end_idx = min(idx + batch_size, n_vocs)
                batch = vocalizations[idx:end_idx]
                idx += batch.shape[0]
                if batch.shape[1] < pad_len:
                    temp = np.zeros((batch.shape[0], batch.shape[1], pad_len), dtype=batch.dtype)
                    temp[..., :batch.shape[-1]] = batch
                    batch = temp
                elif batch.shape[1] > pad_len:
                    batch = batch[..., :pad_len]
                
                batch = torch.from_numpy(batch)
                if use_gpu:
                    batch = batch.cuda()
                yield self.model(batch).detach().cpu().numpy()

    def _setup_output_dirs(self):
        # Save results to `../../trained_models/{config}/{job_id}` directory.
        self.output_dir = os.path.join(
            base_dir,
            "trained_models",
            self._config['CONFIG_NAME'],
            "{0:05g}".format(self._job_id)
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.sample_output_dir = os.path.join(
            self.output_dir,
            'val_predictions'
        )
        os.makedirs(self.sample_output_dir, exist_ok=True)

        # Write the active configuration to disk
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as ctx:
            json.dump(self._config, ctx, indent=4)

        # Configure log file.
        log_filepath = os.path.join(self.output_dir, 'train_log.txt')
        ch = logging.FileHandler(filename=log_filepath, mode='w')
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        self.progress_log = ProgressLogger(
            self._config["NUM_EPOCHS"],
            self.traindata,
            self.valdata,
            self._config["LOG_INTERVAL"],
            self.output_dir,
            logger
        )
        print(f"Saving logs to file: `{log_filepath}`")

        # Create `./results/{job id}/ckpts/{ckpt id}` folders for model weights.
        self.best_weights_file = os.path.join(self.output_dir, "best_weights.pt")
        self.init_weights_file = os.path.join(self.output_dir, "init_weights.pt")
        self.final_weights_file = os.path.join(self.output_dir, "final_weights.pt")
    
    def save_weights(self, weight_path):
        torch.save(self.model.state_dict(), weight_path)

    def _setup_model(self):
        """ Creates the model, optimizer, and loss function.
        """
        # Set random seeds. Note that numpy random seed will affect
        # the data augmentation under the current implementation.
        torch.manual_seed(self._config["TORCH_SEED"])
        np.random.seed(self._config["NUMPY_SEED"])

        # Specify network architecture and loss function.
        self.model, self._loss_fn = build_model(self._config)
        if not self._eval:
            logger.info(self.model.__repr__())

            # Specify optimizer.
            self.optim = torch.optim.SGD(
                self.model.parameters(),
                lr=0.0, # this is set manually at the start of each epoch.
                momentum=self._config["MOMENTUM"],
                weight_decay=self._config["WEIGHT_DECAY"]
            )

    def _query_mem_usage(self):
        used_gb = torch.cuda.max_memory_allocated() / (2**30)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (2**30)
        torch.cuda.reset_max_memory_allocated()
        return 'Max mem. usage: {:.2f}/{:.2f}GiB'.format(used_gb, total_gb)

    def _setup_dataloaders(self):
        # Load training set, validation set, test set.
        self.traindata, self.valdata, self.testdata = build_dataloaders(
            self._datafile, self._config
        )
        logger.info(f"Training set:\t{self.traindata.dataset}")
        logger.info(f"Validation set:\t{self.valdata.dataset}")
        logger.info(f"Test set:\t{self.testdata.dataset}")


    def _get_lr(self, epoch_num: int) -> float:
        """
        Given integer `epoch` specifying the current training
        epoch, return learning rate.
        """
        lm0 = self._config["MIN_LEARNING_RATE"]
        lm1 = self._config["MAX_LEARNING_RATE"]
        f = epoch_num / self._config["NUM_EPOCHS"]
        return (
            lm0 + 0.5 * (lm1 - lm0) * (1 + np.cos(f * np.pi))
        )
    
    @classmethod
    def from_trained_model(cls,
        config: JSON, *,
        model_name: Optional[str]=None,
        job_id: Optional[int]=None,
        device_override: Optional[str]=None
    ):
        """ Initializes the model with weights initialized from the best weights of a
        previously trained instance.
        Parameters:
        config (JSON):
            - config dictionary for the model
        model_name (str):
            - Model name under which the weights will be located. If not provided,
            will be sourced from config dict (assuming it's present)
        job_id (int):
            - Job ID in which the weights will be located. If not provided, will be
            sourced from config dict (assuming it's present)
        device_override (str):
            - When specified, overrides the device (cpu or gpu) present in the config
        """
        if job_id is None:
            if config['JOB_ID'] is None:
                raise ValueError('No job id provided. Cannot locate weights.')
            job_id = config['JOB_ID']
        if model_name is None:
            if config['CONFIG_NAME'] is None:
                raise ValueError('No model name provided. Cannot locate weights')
            model_name = config['CONFIG_NAME']
        
        if device_override is not None:
            device = device_override
            # This needs to be changed so the model is loaded into the right device
            config['DEVICE'] = device.upper()
        else:
            device = config['DEVICE'].lower()
        
        device = 'cuda:0' if device == 'gpu' else 'cpu'

        trainer = cls(None, config, job_id, eval_mode=True)

        weight_file = os.path.join(
            base_dir,
            "trained_models",
            model_name,
            "{0:05g}".format(job_id),
            "best_weights.pt"
        )

        weights = torch.load(weight_file, map_location=device)
        trainer.model.load_state_dict(weights)

        return trainer


def run():
    args = get_args()
    trainer = Trainer(args.datafile, args.config_data, args.job_id)

    num_epochs = args.config_data['NUM_EPOCHS']
    for epoch in range(num_epochs):
        trainer.train_epoch(epoch)
        trainer.eval_validation()

    logger.info(
        ">> FINISHED ALL EPOCHS. TOTAL TIME ELAPSED: " +
        trainer.progress_log.print_time_since_initialization()
    )
    logger.info(f">> SAVING FINAL MODEL WEIGHTS TO {trainer.final_weights_file}")
    trainer.save_weights(trainer.final_weights_file)


if __name__ == '__main__':
    run()

# TODO: Reintroduce test loss after rewriting non-mse loss functions to produce losses
# in real world units
"""
# === EVAL TEST ACCURACY === #
logger.info(f">> EVALUATING ACCURACY ON TEST SET...")

total_testloss = 0.0
model.eval()
with torch.no_grad():
    for sounds, locations in testdata:

        # Move data to gpu
        if CONFIG["DEVICE"] == "GPU":
            sounds = sounds.cuda()
            locations = locations.cuda()

        # Forward pass.
        outputs = model(sounds)

        # Compute loss.
        losses = loss_function(outputs, locations)
        total_testloss += torch.sum(losses).item()

# Write test loss to file
with open(os.path.join(output_dir, "test_set_rmse.txt"), "w") as f:
    f.write(f"{1e3 * np.sqrt(total_testloss / len(testdata))}\n")
"""