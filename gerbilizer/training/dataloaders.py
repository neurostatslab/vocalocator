"""
Functions to construct Datasets and DataLoaders for training and inference
"""

import os
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


class GerbilVocalizationDataset(Dataset):
    def __init__(
        self,
        datapath: str,
        *,
        crop_length: int = 8192,
        inference: bool = False,
        arena_dims: Optional[Tuple[float, float]] = None,
    ):
        """A dataloader designed to return batches of vocalizations with similar lengths

        Args:
            datapath (str): Path to HDF5 dataset
            inference (bool, optional): When true, labels will be returned in addition to data. Defaults to False.
            crop_length (int, optional): When provided, will serve random crops of fixed length instead of full vocalizations
        """
        if isinstance(datapath, str):
            self.dataset = h5py.File(datapath, "r")
        else:
            self.dataset = datapath

        if "len_idx" not in self.dataset:
            raise ValueError("Improperly formatted dataset")

        self.inference = inference
        self.arena_dims = arena_dims
        self.crop_length = crop_length

    def __len__(self):
        return len(self.dataset["len_idx"]) - 1

    def __getitem__(self, idx):
        return self.__processed_data_for_index__(idx)

    def __del__(self):
        self.dataset.close()

    @staticmethod
    def __make_crop(audio: torch.Tensor, crop_length: int):
        """Given an audio sample of shape (n_samples, n_channels), return a random crop
        of shape (crop_length, n_channels)
        """
        audio_len, _ = audio.shape
        if crop_length is None:
            raise ValueError("Cannot take crop without crop length")
        valid_range = audio_len - crop_length
        if valid_range <= 0:  # Audio is shorter than desired crop length, pad right
            pad_size = crop_length - audio_len
            return F.pad(audio, (0, 0, 0, pad_size))
        range_start = np.random.randint(0, valid_range)
        range_end = range_start + crop_length
        return audio[range_start:range_end, :]

    def __audio_for_index(self, dataset: h5py.File, idx: int):
        """Gets an audio sample from the dataset. Will determine the format
        of the dataset and handle it appropriately.
        """
        start, end = dataset["len_idx"][idx : idx + 2]
        audio = dataset["vocalizations"][start:end, ...]
        return audio

    def __label_for_index(self, dataset: h5py.File, idx: int):
        return dataset["locations"][idx]

    @staticmethod
    def scale_features(
        inputs: np.ndarray,
        labels: np.ndarray,
        arena_dims: Tuple[int, int],
    ):
        """Scales the inputs to have zero mean and unit variance. Labels are scaled
        from millimeter units to an arbitrary unit with range [-1, 1].
        """

        scaled_labels = None
        if labels is not None and arena_dims is not None:
            scaled_labels = np.empty_like(labels)

            # Shift range to [-1, 1]
            x_scale = arena_dims[0] / 2  # Arena half-width (mm)
            y_scale = arena_dims[1] / 2
            scaled_labels = labels / np.array([x_scale, y_scale])

        scaled_audio = np.empty_like(inputs)
        # std scaling: I think it's ok to use sample statistics instead of population statistics
        # because we treat each vocalization independantly of the others, their scale w.r.t other
        # vocalizations shouldn't affect our task
        raw_audio_mean = inputs.mean(axis=(-2, -1), keepdims=True)
        raw_audio_std = inputs.std(axis=(-2, -1), keepdims=True)
        scaled_audio = (inputs - raw_audio_mean) / raw_audio_std

        return scaled_audio, scaled_labels

    @staticmethod
    def unscale_features(
        labels: Union[np.ndarray, torch.Tensor],
        arena_dims: Union[Tuple[int, int], np.ndarray, torch.Tensor],
    ):
        """Changes the units of `labels` from arb. scaled unit (in range [-1, 1]) to
        centimeters.
        """
        if not any(
            [isinstance(arena_dims, torch.Tensor), isinstance(arena_dims, np.ndarray)]
        ):
            scale = np.array(arena_dims) / 2
        else:
            scale = arena_dims / 2
        scaled_labels = labels * scale
        return scaled_labels

    def __processed_data_for_index__(self, idx: int):
        sound = self.__audio_for_index(self.dataset, idx).astype(np.float32)
        sound = torch.from_numpy(sound)
        sound = self.__make_crop(sound, self.crop_length)
        location = None if self.inference else self.__label_for_index(self.dataset, idx)

        arena_dims = np.array(self.arena_dims)
        sound, location = GerbilVocalizationDataset.scale_features(
            sound, location, arena_dims=arena_dims
        )

        if self.inference:
            return sound
        return sound, location


def build_dataloaders(path_to_data: str, config: dict):
    # Construct Dataset objects.
    train_path = os.path.join(path_to_data, "train_set.h5")
    val_path = os.path.join(path_to_data, "val_set.h5")
    test_path = os.path.join(path_to_data, "test_set.h5")

    arena_dims = config["DATA"]["ARENA_DIMS"]
    batch_size = config["DATA"]["BATCH_SIZE"]
    crop_length = config["DATA"].get("CROP_LENGTH", None)

    avail_cpus = max(1, len(os.sched_getaffinity(0)) - 1)

    if os.path.exists(train_path):
        traindata = GerbilVocalizationDataset(
            train_path,
            arena_dims=arena_dims,
            crop_length=crop_length,
        )
        train_dataloader = DataLoader(
            traindata, num_workers=avail_cpus, batch_size=batch_size, shuffle=True
        )
    else:
        train_dataloader = None

    if os.path.exists(val_path):
        valdata = GerbilVocalizationDataset(
            val_path,
            arena_dims=arena_dims,
            crop_length=crop_length,
        )
        val_dataloader = DataLoader(valdata, batch_size=batch_size, shuffle=False)
    else:
        val_dataloader = None

    if os.path.exists(test_path):
        testdata = GerbilVocalizationDataset(
            test_path,
            arena_dims=arena_dims,
            crop_length=crop_length,
            inference=True,
        )
        test_dataloader = DataLoader(testdata, batch_size=batch_size, shuffle=False)
    else:
        test_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader
