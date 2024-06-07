"""
Functions to construct Datasets and DataLoaders for training and inference
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import torch
from scipy.io import wavfile
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchaudio import functional as AF


class VocalizationDataset(Dataset):
    def __init__(
        self,
        datapath: Union[Path, str],
        crop_length: int,
        *,
        inference: bool = False,
        arena_dims: Optional[Union[np.ndarray, Tuple[float, float]]] = None,
        index: Optional[np.ndarray] = None,
        normalize_data: bool = True,
        sample_rate: int = 192000,
        sample_vocalization_dir: Optional[Path] = None,
    ):
        """
        Args:
            datapath (Path): Path to HDF5 dataset
            inference (bool, optional): When true, data will be cropped deterministically. Defaults to False.
            crop_length (int): Length of audio samples to return.
            arena_dims (Optional[Union[np.ndarray, Tuple[float, float]]], optional): Dimensions of the arena in mm. Used to scale labels.
            index (Optional[np.ndarray], optional): An array of indices to use for this dataset. Defaults to None, which will use the full dataset
        """
        if isinstance(datapath, str):
            datapath = Path(datapath)
        self.datapath = datapath
        dataset = h5py.File(self.datapath, "r")

        # dataset cannot exist as a member of the object until after pytorch has cloned and
        # spread the dataset objects across multiple processes.
        # This is because h5py handles cannot be pickled and pytorch uses pickle under the hood
        # I get around this by re-initializing the h5py.File lazily in __getitem__
        self.dataset: Optional[h5py.File] = None

        if not isinstance(arena_dims, np.ndarray):
            arena_dims = np.array(arena_dims).astype(np.float32)

        if "length_idx" not in dataset and "rir_length_idx" not in dataset:
            raise ValueError("Improperly formatted dataset")

        if "audio" in dataset:
            self.is_rir_dataset = False
            self.length = len(dataset["length_idx"]) - 1
        elif "rir" in dataset:
            self.is_rir_dataset = True
            self.length = len(dataset["rir_length_idx"]) - 1
        else:
            raise ValueError("Improperly formatted dataset")

        self.inference = inference
        self.arena_dims = arena_dims
        self.crop_length = crop_length
        self.index = index
        self.normalize_data = normalize_data
        self.sample_rate = sample_rate
        self.sample_vocalization_dir = sample_vocalization_dir

        if self.index is not None:
            self.length = len(self.index)

        if self.is_rir_dataset and self.sample_vocalization_dir is None:
            raise ValueError("RIR dataset requires sample vocalizations")

        worker_info = torch.utils.data.get_worker_info()
        seed = 0 if worker_info is None else worker_info.seed
        self.rng = np.random.default_rng(seed)

        self.sample_vocalizations = []
        if self.is_rir_dataset:
            for wavfile_path in Path(self.sample_vocalization_dir).glob("*.wav"):
                fs, data = wavfile.read(wavfile_path)
                if fs != sample_rate:
                    continue
                data = self.convert_audio_to_float(data)
                if data is None:
                    continue
                if len(data.shape) > 1:
                    # scipy reads stereo files as (n_samples, n_channels)
                    channel_powers = np.sum(data**2, axis=0)
                    data = data[:, np.argmax(channel_powers)]
                # Don't want some vocalizations to be louder than others
                data = (data - data.mean()) / data.std()
                self.sample_vocalizations.append(torch.from_numpy(data).float())

            if not self.sample_vocalizations:
                raise ValueError("No valid vocalizations found")
        dataset.close()

    def convert_audio_to_float(self, audio: np.ndarray):
        """Converts audio to float32 and scales it to the range [-1, 1]"""
        float_types = (np.float32, np.float64)
        if audio.dtype in float_types:
            return audio.astype(np.float32)

        int_types = (np.int8, np.int16, np.int32)
        if audio.dtype not in int_types:
            return None
        iinfo = np.iinfo(audio.dtype)
        imin, imax = iinfo.min, iinfo.max
        return (audio.astype(np.float64) / (imax - imin)).astype(np.float32)

    def __len__(self):
        return self.length

    def __del__(self):
        if self.dataset is not None:
            self.dataset.close()

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.datapath, "r")
        true_idx = idx
        if self.index is not None:
            true_idx = self.index[idx]
        return self.__processed_data_for_index__(true_idx)

    def __make_crop(self, audio: torch.Tensor, crop_length: int):
        """Given an audio sample of shape (n_samples, n_channels), return a random crop
        of shape (crop_length, n_channels)
        """
        audio_len, _ = audio.shape
        valid_range = audio_len - crop_length
        if valid_range <= 0:  # Audio is shorter than desired crop length, pad right
            pad_size = crop_length - audio_len
            # will fail if input is numpy array
            return F.pad(audio, (0, 0, 0, pad_size))
        if self.inference:
            range_start = 0
        else:
            range_start = self.rng.integers(0, valid_range)
        range_end = range_start + crop_length
        return audio[range_start:range_end, :]

    def __rir_for_index(self, idx: int):
        if self.is_rir_dataset:
            start, end = self.dataset["rir_length_idx"][idx : idx + 2]
            return torch.from_numpy(self.dataset["rir"][start:end, ...]).float()
        raise ValueError("Dataset does not contain RIRs")

    def __audio_for_index(self, idx: int):
        """Gets an audio sample from the dataset. Will determine the format
        of the dataset and handle it appropriately.
        """
        start, end = self.dataset["length_idx"][idx : idx + 2]
        audio = self.dataset["audio"][start:end, ...]
        return torch.from_numpy(audio.astype(np.float32))

    def __label_for_index(self, idx: int):
        if "locations" not in self.dataset:
            return None
        return torch.from_numpy(self.dataset["locations"][idx].astype(np.float32))

    def scale_features(
        self,
        audio: np.ndarray,
        labels: np.ndarray,
    ):
        """Scales the inputs to have zero mean and unit variance. Labels are scaled
        from millimeter units to an arbitrary unit with range [-1, 1].
        """

        scaled_labels = None
        if labels is not None and self.arena_dims is not None:
            # Shift range to [-1, 1]
            scaled_labels = labels
            scaled_labels[..., : len(self.arena_dims)] /= (
                torch.from_numpy(self.arena_dims).float() / 2
            )

        if self.normalize_data:
            scaled_audio = (audio - audio.mean()) / audio.std()

        return scaled_audio, scaled_labels

    def __processed_data_for_index__(self, idx: int):
        if self.is_rir_dataset:
            # rir shape: (n_channels, n_samples)
            rir = self.__rir_for_index(idx)
            # sample_vocalization shape: (n_samples,)
            sample_vocalization = self.sample_vocalizations[
                self.rng.integers(len(self.sample_vocalizations))
            ]

            sound = AF.convolve(rir.T, sample_vocalization[None, :], mode="full").T
        else:
            sound = self.__audio_for_index(idx)

        sound = self.__make_crop(sound, self.crop_length)
        location = self.__label_for_index(idx)

        sound, location = self.scale_features(sound, location)

        return sound, location

    def collate(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate function for the dataloader. Takes a list of (audio, label) tuples and returns
        a batch of audio and labels.
        """
        audio, labels = [x[0] for x in batch], [x[1] for x in batch]
        audio = torch.stack(audio)
        if labels[0] is not None:
            labels = torch.stack(labels)
        else:
            labels = [None] * len(audio)
        return audio, labels


def build_dataloaders(
    path_to_data: Union[Path, str], config: dict, index_dir: Optional[Path]
):
    # Construct Dataset objects.
    if not isinstance(path_to_data, Path):
        path_to_data = Path(path_to_data)

    arena_dims = config["DATA"]["ARENA_DIMS"]
    batch_size = config["DATA"]["BATCH_SIZE"]
    crop_length = config["DATA"]["CROP_LENGTH"]
    normalize_data = config["DATA"].get("NORMALIZE_DATA", True)
    sample_rate = config["DATA"].get("SAMPLE_RATE", 192000)

    vocalization_dir = config["DATA"].get(
        "VOCALIZATION_DIR",
        None,
    )

    index_arrays = {"train": None, "val": None}
    if index_dir is not None:
        index_arrays["train"] = np.load(index_dir / "train_set.npy")
        index_arrays["val"] = np.load(index_dir / "val_set.npy")

    try:
        avail_cpus = max(1, len(os.sched_getaffinity(0)) - 1)
    except:
        avail_cpus = max(1, os.cpu_count() - 1)

    if path_to_data.is_dir():
        train_path = path_to_data / "train_set.h5"
        val_path = path_to_data / "val_set.h5"
    else:
        train_path = path_to_data
        val_path = path_to_data
        if index_dir is None:
            # manually create train/val split
            with h5py.File(train_path, "r") as f:
                if "length_idx" in f:
                    dset_size = len(f["length_idx"]) - 1
                elif "rir_length_idx" in f:
                    dset_size = len(f["rir_length_idx"]) - 1
                else:
                    raise ValueError("Improperly formatted dataset")
            full_index = np.arange(dset_size)
            rng = np.random.default_rng(0)
            rng.shuffle(full_index)
            index_arrays["train"] = full_index[: int(0.8 * dset_size)]
            index_arrays["val"] = full_index[
                int(0.8 * dset_size) : int(0.9 * dset_size)
            ]

    traindata = VocalizationDataset(
        train_path,
        arena_dims=arena_dims,
        crop_length=crop_length,
        index=index_arrays["train"],
        normalize_data=normalize_data,
        sample_rate=sample_rate,
        sample_vocalization_dir=vocalization_dir,
    )

    valdata = VocalizationDataset(
        val_path,
        arena_dims=arena_dims,
        crop_length=crop_length,
        inference=True,
        index=index_arrays["val"],
        normalize_data=normalize_data,
        sample_rate=sample_rate,
        sample_vocalization_dir=vocalization_dir,
    )

    train_dataloader = DataLoader(
        traindata,
        batch_size=batch_size,
        shuffle=True,
        num_workers=avail_cpus,
        collate_fn=traindata.collate,
    )

    val_dataloader = DataLoader(
        valdata,
        batch_size=batch_size,
        num_workers=avail_cpus,
        shuffle=False,
        collate_fn=valdata.collate,
    )

    return train_dataloader, val_dataloader
