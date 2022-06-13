"""
Functions to construct torch Dataset, DataLoader
objects and specify data augmentation.
"""

import os
import h5py
import numpy as np
from torch import randint
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d


class GerbilVocalizationDataset(Dataset):
    def __init__(
        self, datapath, *,
        flip_vert=False, flip_horiz=False,
        segment_len=256
    ):
        """
        Args:
            datapath (str):
                Path to directory containing the 'snippet{idx}' subdirectories
            flip_vert (bool):
                When true, mirroring augmentation will be applied to data and labels
            flip_horiz (bool):
                When true, mirroring augmentation will be applied to data and labels
        """
        self.datapath = datapath
        self.dataset = h5py.File(datapath, 'r')
        self.flip_vert = flip_vert
        self.flip_horiz = flip_horiz
        self.segment_len = segment_len
        self.samp_size = 1

    def __del__(self):
        self.dataset.close()

    def __len__(self):
        if 'len_idx' in self.dataset:
            return len(self.dataset['len_idx']) - 1
        return len(self.dataset['vocalizations'])

    def _crop_audio(self, audio):
        # TODO: Delete this fn?
        pad_len = self.crop_audio
        n_samples = audio.shape[0]
        n_channels = audio.shape[1]

        new_len = min(pad_len, n_samples)
        clipped_audio = audio[:new_len, ...].T  # Change shape to (n_mics, <=pad_len)

        zeros = np.zeros((n_channels, pad_len), dtype=audio.dtype)
        zeros[:, :clipped_audio.shape[1]] = clipped_audio
        return zeros
    
    @classmethod
    def sample_segment(cls, audio, section_len):
        """ Samples a contiguous segment of length `section_len` from audio sample `audio` randomly
        within margins extending 10% of the total audio length from either end of the audio sample.

        Returns: audio segment with shape (n_channels, section_len)
        """
        n_samp = len(audio)
        margin = int(n_samp * 0.1)
        idx_range = margin, n_samp-margin-section_len
        if n_samp - 2*margin <= section_len:
            # If section_len is longer than the audio we're sampling from, randomly place the entire
            # audio sample within a block of zeros of length section_len
            padding = np.zeros((audio.shape[1], section_len))
            offset = randint(-margin, margin, (1,)).item()
            end = min(audio.shape[0] + offset, section_len)
            if offset < 0:
                padding[:, :end] = audio[-offset:end-offset, :].T
            else:
                padding[:, offset:end] = audio[:end-offset, :].T
            return padding
        start = randint(*idx_range, (1,)).item()
        end = start + section_len
        return audio[start:end, ...].T
    
    def _audio_for_index(self, dataset, idx):
        """ Gets an audio sample from the dataset. Will determine the format
        of the dataset and handle it appropriately.
        """
        if 'len_idx' in dataset:
            start, end = dataset['len_idx'][idx:idx+2]
            audio = dataset['vocalizations'][start:end, ...]
            return audio
        else:
            return dataset['vocalizations'][idx]
    
    @classmethod
    def scale_features(cls, inputs, labels, is_batch=False):
        """ Scales the inputs to have zero mean and unit variance. Labels are scaled
        from millimeter units to an arbitrary unit with range [0, 1].
        """

        if labels is not None:
            scaled_labels = np.empty_like(labels)

            is_map = len(scaled_labels.shape) == (3 if is_batch else 2)
            if not is_map:
                # Shift range to [-1, 1]
                scaled_labels[..., 0] = labels[..., 0] / 300
                scaled_labels[..., 1] = labels[..., 1] / 200
            else:
                if is_batch:
                    lmin = labels.min(axis=(-1, -2), keepdims=True)
                    lmax = labels.max(axis=(-1, -2), keepdims=True)
                else:
                    lmin = labels.min()
                    lmax = labels.max()
                scaled_labels = (labels - lmin) / (lmax - lmin)
        else:
            scaled_labels = None

        scaled_audio = np.empty_like(inputs)
        # std scaling: I think it's ok to use sample statistics instead of population statistics
        # because we treat each vocalization independantly of the others, their scale w.r.t other
        # vocalizations shouldn't affect our task
        raw_audio_mean = inputs[..., :4, :].mean()
        raw_audio_std = inputs[..., :4, :].std()
        scaled_audio[..., :4, :] = (inputs[..., :4, :] - raw_audio_mean) / raw_audio_std
        if (inputs.shape[1] if is_batch else inputs.shape[0]) == 10:
            xcorr_mean = inputs[..., 4:, :].mean()
            xcorr_std = inputs[..., 4:, :].std()
            scaled_audio[..., 4:, :] = (inputs[..., 4:, :] - xcorr_mean) / xcorr_std
        
        return scaled_audio, scaled_labels

    @classmethod
    def unscale_features(cls, labels):
        """ Changes the units of `labels` from arb. scaled unit (in range [0, 1]) to
        centimeters.
        """
        scaled_labels = np.empty_like(labels)
        scaled_labels[..., 0] = labels[..., 0] * 30
        scaled_labels[..., 1] = labels[..., 1] * 20
        return scaled_labels

    def __getitem__(self, idx):
        
        # Load audio waveforms in time domain. Each sample is held
        # in a matrix with dimensions (10, num_audio_samples)
        #
        # The four microphones are arranged like this:
        #
        #           1-------------0
        #           |             |
        #           |             |
        #           |             |
        #           |             |
        #           2-------------3
        #
        # The 
        # 0 - mic 0 trace
        # 1 - mic 1 trace
        # 2 - mic 2 trace
        # 3 - mic 3 trace
        # 4 - (0, 1) - cross-correlation of mic 0 and mic 1
        # 5 - (0, 2) - cross-correlation of mic 0 and mic 2
        # 6 - (0, 3) - cross-correlation of mic 0 and mic 3
        # 7 - (1, 2) - cross-correlation of mic 1 and mic 2
        # 8 - (1, 3) - cross-correlation of mic 1 and mic 3
        # 9 - (2, 3) - cross-correlation of mic 2 and mic 3
        #
        sound = self._audio_for_index(self.dataset, idx)
    
        if self.samp_size > 1:
            sound = np.stack([self.sample_segment(sound, self.segment_len) for _ in range(self.samp_size)], axis=0)
        else:
            sound = self.sample_segment(sound, self.segment_len)

        # Load animal location in the environment.
        # shape: (2 (x/y coordinates), )
        location_map = self.dataset['locations'][idx][:]
        sound, location_map = GerbilVocalizationDataset.scale_features(sound, location_map, is_batch=self.samp_size>1)

        is_map = len(location_map.shape) == 2

        # With p = 0.5, flip vertically
        if self.flip_vert and np.random.binomial(1, 0.5):
            # Assumes the center of the enclosure is (0, 0)
            if is_map:
                location_map = location_map[::-1, :]
            else:
                location_map[1] *= -1
            # mic 0 -> mic 3
            # mic 1 -> mic 2
            # mic 2 -> mic 1
            # mic 3 -> mic 0
            # (0, 1) -> (3, 2)  so  4 -> 9
            # (0, 2) -> (3, 1)  so  5 -> 8
            # (0, 3) -> (3, 0)  so  6 -> 6
            # (1, 2) -> (2, 1)  so  7 -> 7
            # (1, 3) -> (2, 0)  so  8 -> 5
            # (2, 3) -> (1, 0)  so  9 -> 4
            if sound.shape[0] == 10:
                sound = sound[[3, 2, 1, 0, 9, 8, 6, 7, 5, 4]]
            else:
                sound = sound[[3, 2, 1, 0]]

        # With p = 0.5, flip horizontally
        if self.flip_horiz and np.random.binomial(1, 0.5):
            # Assumes the center of the enclosure is (0, 0)
            if is_map:
                location_map = location_map[:, ::-1]
            else:
                location_map[0] *= -1
            # mic 0 -> mic 1
            # mic 1 -> mic 0
            # mic 2 -> mic 3
            # mic 3 -> mic 2
            # (0, 1) -> (1, 0)  so  4 -> 4
            # (0, 2) -> (1, 3)  so  5 -> 8
            # (0, 3) -> (1, 2)  so  6 -> 7
            # (1, 2) -> (0, 3)  so  7 -> 6
            # (1, 3) -> (0, 2)  so  8 -> 5
            # (2, 3) -> (3, 2)  so  9 -> 9
            if sound.shape[0] == 10:
                sound = sound[[1, 0, 3, 2, 4, 8, 7, 6, 5, 9]]
            else:
                sound = sound[[1, 0, 3, 2]]

        return sound.astype("float32"), location_map.astype("float32")


def build_dataloaders(path_to_data, CONFIG):

    # Construct Dataset objects.
    traindata = GerbilVocalizationDataset(
        os.path.join(path_to_data, "train_set.h5"),
        flip_vert=(CONFIG["AUGMENT_LABELS"] and CONFIG["AUGMENT_FLIP_VERT"]),
        flip_horiz=(CONFIG["AUGMENT_LABELS"] and CONFIG["AUGMENT_FLIP_HORIZ"])
    )
    # TODO -- make new validation and test set files!
    valdata = GerbilVocalizationDataset(
        os.path.join(path_to_data, "val_set.h5"),
        flip_vert=False, flip_horiz=False
    )
    testdata = GerbilVocalizationDataset(
        os.path.join(path_to_data, "test_set.h5"),
        flip_vert=False, flip_horiz=False
    )

    # Construct DataLoader objects.
    train_dataloader = DataLoader(
        traindata,
        batch_size=CONFIG["TRAIN_BATCH_SIZE"],
        shuffle=True
    )
    val_dataloader = DataLoader(
        valdata,
        batch_size=CONFIG["VAL_BATCH_SIZE"],
        shuffle=False
    )
    test_dataloader = DataLoader(
        testdata,
        batch_size=CONFIG["TEST_BATCH_SIZE"],
        shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader






