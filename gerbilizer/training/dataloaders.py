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


class GerbilRIRDataset(Dataset):
    def __init__(
        self,
        rir_dataset_path: Path,
        sample_vocalization_dir: Path,
        crop_length: int,
        *,
        inference: bool = False,
        arena_dims: Optional[Union[np.ndarray, Tuple[float, float]]] = None,
        index: Optional[np.ndarray] = None,
        normalize_data: bool = True,
    ):
        """Construct a dataset from a set of Room Impulse Responses (RIRs) and audio to be played through them.
        The audio should be provided as a directory of wav files at 125kHz. The RIR dataset should be a path to
        an HDF5 file with the following structure:
          - rir: A dataset of shape (n_samples, n_channels) containing the concatenated, padded RIRs for each channel
          - locations: A dataset of shape (n_samples, 3) containing the locations of the sound source in the RIR dataset
          - rir_length_idx: A dataset of shape (n_samples + 1) containing the indices of the start and end of each RIR in the rir dataset
        """
        self.rir_dataset = h5py.File(rir_dataset_path, "r")
        self.sample_vocalization_dir = sample_vocalization_dir
        self.normalize_data = normalize_data
        # Load sample vocalizations
        self.sample_vocalizations = []
        for wavfile_path in self.sample_vocalization_dir.glob("*.wav"):
            fs, data = wavfile.read(wavfile_path)
            if fs != 125000:
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

        self.crop_length = crop_length
        self.inference = inference
        self.arena_dims = torch.tensor(arena_dims).float()
        self.index = index

        # Make a random number generator
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        self.rng = np.random.default_rng(worker_id)

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

    def __del__(self):
        self.rir_dataset.close()

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        return len(self.rir_dataset["rir_length_idx"]) - 1

    def __getitem__(self, idx):
        if self.index is not None:
            idx = self.index[idx]
        return self.__processed_data_for_index__(idx)

    def __rir_for_index(self, idx):
        """Gets an RIR sample from the dataset. Will determine the format
        of the dataset and handle it appropriately.
        """
        start, end = self.rir_dataset["rir_length_idx"][idx : idx + 2]
        rir = self.rir_dataset["rir"][start:end, ...].T
        return torch.from_numpy(rir).float()

    def __label_for_index(self, idx):
        if "locations" not in self.rir_dataset:
            return None
        return torch.from_numpy(self.rir_dataset["locations"][idx, :2]).float()

    def __make_crop(self, audio: torch.Tensor):
        """Given an audio sample of shape (n_samples, n_channels), return a random crop
        of shape (crop_length, n_channels)
        """
        crop_length = self.crop_length
        audio_len, _ = audio.shape
        valid_range = audio_len - crop_length
        if valid_range <= 0:
            pad_size = crop_length - audio_len
            return F.pad(audio, (0, 0, 0, pad_size))
        if self.inference:
            range_start = 0
        else:
            range_start = np.random.randint(0, valid_range)
        range_end = range_start + crop_length
        return audio[range_start:range_end, :]

    def scale_features(
        self,
        audio: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Scales the inputs to have zero mean and unit variance. Labels are scaled
        from millimeter units to an arbitrary unit with range [-1, 1].
        """

        if labels is not None and self.arena_dims is not None:
            half_arena_dims = self.arena_dims / 2
            labels = labels / half_arena_dims

        if self.normalize_data:
            audio = (audio - audio.mean()) / audio.std()
        return audio, labels

    def __processed_data_for_index__(self, idx: int):
        # rir: (n_channels, n_samples)
        rir = self.__rir_for_index(idx)
        # sample_vocalization: (n_samples,)
        sample_vocalization = self.sample_vocalizations[
            self.rng.integers(len(self.sample_vocalizations))
        ]

        audio = AF.convolve(rir, sample_vocalization[None, :], mode="full")
        # After the following line, audio is (n_samples, n_channels)
        audio = self.__make_crop(audio.T)

        location = self.__label_for_index(idx)
        audio, location = self.scale_features(audio, location)
        return audio, location

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


class GerbilVocalizationDataset(Dataset):
    def __init__(
        self,
        datapath: Union[Path, str],
        crop_length: int,
        *,
        inference: bool = False,
        arena_dims: Optional[Union[np.ndarray, Tuple[float, float]]] = None,
        index: Optional[np.ndarray] = None,
        normalize_data: bool = True,
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
        if isinstance(datapath, Path):
            self.dataset = h5py.File(datapath, "r")
        else:
            self.dataset = datapath

        if not isinstance(arena_dims, np.ndarray):
            arena_dims = np.array(arena_dims).astype(np.float32)

        if "length_idx" not in self.dataset:
            raise ValueError("Improperly formatted dataset")

        self.inference = inference
        self.arena_dims = arena_dims
        self.crop_length = crop_length
        self.index = index
        self.normalize_data = normalize_data

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        return len(self.dataset["length_idx"]) - 1

    def __getitem__(self, idx):
        if self.index is not None:
            idx = self.index[idx]
        return self.__processed_data_for_index__(idx)

    def __del__(self):
        self.dataset.close()

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
            range_start = np.random.randint(0, valid_range)
        range_end = range_start + crop_length
        return audio[range_start:range_end, :]

    def __audio_for_index(self, dataset: h5py.File, idx: int):
        """Gets an audio sample from the dataset. Will determine the format
        of the dataset and handle it appropriately.
        """
        start, end = dataset["length_idx"][idx : idx + 2]
        audio = dataset["audio"][start:end, ...]
        return torch.from_numpy(audio).float()

    def __label_for_index(self, dataset: h5py.File, idx: int):
        if "locations" not in dataset:
            return None
        if len(dataset["locations"].shape) == 3:
            return torch.from_numpy(dataset["locations"][idx, 0]).float()
        return torch.from_numpy(dataset["locations"][idx]).float()

    def scale_features(self, audio: torch.Tensor, labels: torch.Tensor):
        """Scales the inputs to have zero mean and unit variance. Labels are scaled
        from millimeter units to an arbitrary unit with range [-1, 1].
        """

        if labels is not None and self.arena_dims is not None:
            # Shift range to [-1, 1]
            labels = labels / torch.from_numpy(self.arena_dims) * 2

        if self.normalize_data:
            audio = (audio - audio.mean()) / audio.std()

        return audio, labels

    def __processed_data_for_index__(self, idx: int):
        sound = self.__audio_for_index(self.dataset, idx)
        sound = self.__make_crop(sound, self.crop_length)

        location = self.__label_for_index(self.dataset, idx)

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

    index_arrays = {"train": None, "val": None}
    if index_dir is not None:
        index_arrays["train"] = np.load(index_dir / "train_set.npy")
        index_arrays["val"] = np.load(index_dir / "val_set.npy")

    avail_cpus = max(1, len(os.sched_getaffinity(0)) - 1)

    if path_to_data.is_dir():
        train_path = path_to_data / "train_set.h5"
        val_path = path_to_data / "val_set.h5"
    else:
        train_path = path_to_data
        val_path = path_to_data

        use_rir_dataset = False
        with h5py.File(train_path, "r") as f:
            if "length_idx" in f:
                use_rir_dataset = False
            elif "rir_length_idx" in f:
                use_rir_dataset = True

        if index_dir is None:
            # manually create train/val split
            with h5py.File(train_path, "r") as f:
                dset_size = (
                    len(f[("rir_length_idx" if use_rir_dataset else "length_idx")]) - 1
                )
            full_index = np.arange(dset_size)
            rng = np.random.default_rng()
            rng.shuffle(full_index)
            index_arrays["train"] = full_index[: int(0.8 * dset_size)]
            index_arrays["val"] = full_index[int(0.8 * dset_size) :]

    if not use_rir_dataset:
        traindata = GerbilVocalizationDataset(
            train_path,
            arena_dims=arena_dims,
            crop_length=crop_length,
            index=index_arrays["train"],
            normalize_data=normalize_data,
        )
        valdata = GerbilVocalizationDataset(
            val_path,
            arena_dims=arena_dims,
            crop_length=crop_length,
            inference=True,
            index=index_arrays["val"],
            normalize_data=normalize_data,
        )
    else:
        traindata = GerbilRIRDataset(
            train_path,
            Path("/mnt/home/atanelus/ceph/good_gerbil_vocalizations"),
            crop_length=crop_length,
            inference=False,
            arena_dims=arena_dims,
            index=index_arrays["train"],
            normalize_data=normalize_data,
        )
        valdata = GerbilRIRDataset(
            val_path,
            Path("/mnt/home/atanelus/ceph/good_gerbil_vocalizations"),
            crop_length=crop_length,
            inference=True,
            arena_dims=arena_dims,
            index=index_arrays["val"],
            normalize_data=normalize_data,
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
