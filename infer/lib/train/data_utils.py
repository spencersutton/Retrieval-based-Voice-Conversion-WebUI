import logging
import os
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.utils.data
from scipy.io import wavfile

from infer.lib.train.mel_processing import spectrogram_torch
from infer.lib.train.params import HParamsData

logger = logging.getLogger(__name__)


def load_wav_to_torch(full_path: Path) -> tuple[torch.FloatTensor, int]:
    sampling_rate, data = cast("tuple[int, np.ndarray]", wavfile.read(full_path))  # pyright: ignore[reportUnknownMemberType]
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename: Path) -> list[list[str]]:
    lines = filename.read_text().splitlines()
    return [line.strip().split("|") for line in lines]


class TextAudioLoaderMultiNSFsid(torch.utils.data.Dataset):
    """
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text: Path, hparams: HParamsData) -> None:
        self.audiopaths_and_text = [
            (Path(x[0]), Path(x[1]), Path(x[2]), Path(x[3]), int(x[4]))
            for x in load_filepaths_and_text(audiopaths_and_text)
        ]
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.lengths = []

    def get_audio_text_pair(
        self, audiopath_and_text: tuple[Path, Path, Path, Path, int]
    ) -> tuple:
        # separate filename and text
        file, phone, pitch, pitchf, dv = audiopath_and_text

        phone, pitch, pitchf = self.get_labels(phone, pitch, pitchf)
        spec, wav = self.get_audio(file)
        dv = torch.LongTensor([int(dv)])

        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]

        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            # amor
            len_wav = len_min * self.hop_length

            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]

            phone = phone[:len_min, :]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]

        return (spec, wav, phone, pitch, pitchf, dv)

    def get_labels(
        self, phone_str: Path, pitch_path: Path, pitchf_path: Path
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        phone = np.load(phone_str)
        phone = np.repeat(phone, 2, axis=0)
        pitch = np.load(pitch_path)
        pitchf = np.load(pitchf_path)
        n_num = min(phone.shape[0], 900)  # DistributedBucketSampler

        phone = phone[:n_num, :]
        pitch = pitch[:n_num]
        pitchf = pitchf[:n_num]
        phone = torch.FloatTensor(phone)
        pitch_tensor = torch.LongTensor(pitch)
        pitchf_tensor = torch.FloatTensor(pitchf)
        return phone, pitch_tensor, pitchf_tensor

    def get_audio(self, filename: str) -> tuple:
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR"
            )
        audio_norm = audio

        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename)
            except Exception:
                logger.warning("%s %s", spec_filename, traceback.format_exc())
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
                spec = torch.squeeze(spec, 0)
                torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        return spec, audio_norm

    def __getitem__(self, index: int) -> tuple:
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self) -> int:
        return len(self.audiopaths_and_text)


class TextAudioCollateMultiNSFsid:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids: bool = False) -> None:
        self.return_ids = return_ids

    def __call__(
        self, batch: list[object]
    ) -> tuple[
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
    ]:
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wave_len = max([x[1].size(1) for x in batch])
        spec_lengths = torch.LongTensor(len(batch))
        wave_lengths = torch.LongTensor(len(batch))
        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        wave_padded = torch.FloatTensor(len(batch), 1, max_wave_len)
        spec_padded.zero_()
        wave_padded.zero_()

        max_phone_len = max([x[2].size(0) for x in batch])
        phone_lengths = torch.LongTensor(len(batch))
        phone_padded = torch.FloatTensor(
            len(batch), max_phone_len, batch[0][2].shape[1]
        )  # (spec, wav, phone, pitch)
        pitch_padded = torch.LongTensor(len(batch), max_phone_len)
        pitchf_padded = torch.FloatTensor(len(batch), max_phone_len)
        phone_padded.zero_()
        pitch_padded.zero_()
        pitchf_padded.zero_()

        sid = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row[0]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wave = row[1]
            wave_padded[i, :, : wave.size(1)] = wave
            wave_lengths[i] = wave.size(1)

            phone = row[2]
            phone_padded[i, : phone.size(0), :] = phone
            phone_lengths[i] = phone.size(0)

            pitch = row[3]
            pitch_padded[i, : pitch.size(0)] = pitch
            pitchf = row[4]
            pitchf_padded[i, : pitchf.size(0)] = pitchf

            sid[i] = row[5]

        return (
            phone_padded,
            phone_lengths,
            pitch_padded,
            pitchf_padded,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
        )


class TextAudioLoader(torch.utils.data.Dataset):
    """
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text: Path, hparams: HParamsData) -> None:
        self.audiopaths_and_text = [
            (Path(x[0]), Path(x[1]), int(x[2]))
            for x in load_filepaths_and_text(audiopaths_and_text)
        ]
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.lengths = []

    def get_audio_text_pair(self, audiopath_and_text: tuple[Path, Path, int]) -> tuple:
        # separate filename and text
        file, phone, dv = audiopath_and_text

        phone = self.get_labels(phone)
        spec, wav = self.get_audio(file)
        dv = torch.LongTensor([int(dv)])

        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]
        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length
            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]
            phone = phone[:len_min, :]
        return (spec, wav, phone, dv)

    def get_labels(self, phone_path: Path) -> torch.FloatTensor:
        phone = np.load(phone_path)
        phone = np.repeat(phone, 2, axis=0)
        n_num = min(phone.shape[0], 900)  # DistributedBucketSampler
        phone = phone[:n_num, :]
        phone = torch.FloatTensor(phone)
        return phone

    def get_audio(self, filename: str) -> tuple:
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR"
            )
        audio_norm = audio

        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename)
            except Exception:
                logger.warning("%s %s", spec_filename, traceback.format_exc())
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
                spec = torch.squeeze(spec, 0)
                torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        return spec, audio_norm

    def __getitem__(self, index: int) -> tuple:
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self) -> int:
        return len(self.audiopaths_and_text)


class TextAudioCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids: bool = False) -> None:
        self.return_ids = return_ids

    def __call__(
        self, batch: list[object]
    ) -> tuple[
        torch.FloatTensor,
        torch.LongTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
    ]:
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wave_len = max([x[1].size(1) for x in batch])
        spec_lengths = torch.LongTensor(len(batch))
        wave_lengths = torch.LongTensor(len(batch))
        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        wave_padded = torch.FloatTensor(len(batch), 1, max_wave_len)
        spec_padded.zero_()
        wave_padded.zero_()

        max_phone_len = max([x[2].size(0) for x in batch])
        phone_lengths = torch.LongTensor(len(batch))
        phone_padded = torch.FloatTensor(
            len(batch), max_phone_len, batch[0][2].shape[1]
        )
        phone_padded.zero_()
        sid = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row[0]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wave = row[1]
            wave_padded[i, :, : wave.size(1)] = wave
            wave_lengths[i] = wave.size(1)

            phone = row[2]
            phone_padded[i, : phone.size(0), :] = phone
            phone_lengths[i] = phone.size(0)

            sid[i] = row[3]

        return (
            phone_padded,
            phone_lengths,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset: object,
        batch_size: int,
        boundaries: list[int],
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self) -> tuple:
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, -1, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self) -> Iterator:
        # deterministically shuffle based on epoch
        g = torch.Generator(device=torch.get_default_device())
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x: int, lo: int = 0, hi: int | None = None) -> int:
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self) -> int:
        return self.num_samples // self.batch_size
