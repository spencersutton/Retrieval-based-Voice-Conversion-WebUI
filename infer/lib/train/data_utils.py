import logging
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
    Loads audio, text, pitch, and pitchf pairs, normalizes and converts them to tensors,
    and computes spectrograms from audio files.
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
        self.lengths = []

    def get_audio_text_pair(self, entry: tuple[Path, Path, Path, Path, int]) -> tuple:
        file, phone_path, pitch_path, pitchf_path, dv = entry
        phone, pitch, pitchf = self.get_labels(phone_path, pitch_path, pitchf_path)
        spec, wav = self.get_audio(file)
        dv_tensor = torch.LongTensor([dv])

        len_min = min(phone.size(0), spec.size(-1))
        if phone.size(0) != spec.size(-1):
            spec = spec[:, :len_min]
            wav = wav[:, : len_min * self.hop_length]
            phone = phone[:len_min, :]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]

        return spec, wav, phone, pitch, pitchf, dv_tensor

    def get_labels(
        self, phone_path: Path, pitch_path: Path, pitchf_path: Path
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        phone = np.load(phone_path)
        pitch = np.load(pitch_path)
        pitchf = np.load(pitchf_path)
        max_len = min(phone.shape[0] * 2, 900)
        phone = np.repeat(phone, 2, axis=0)[:max_len, :]
        pitch = pitch[:max_len]
        pitchf = pitchf[:max_len]
        return (
            torch.FloatTensor(phone),
            torch.LongTensor(pitch),
            torch.FloatTensor(pitchf),
        )

    def get_audio(self, filename: Path) -> tuple[torch.Tensor, torch.Tensor]:
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR"
            )
        audio_norm = audio.unsqueeze(0)
        spec_filename = filename.with_suffix(".spec.pt")
        if spec_filename.exists():
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
    """Zero-pads model inputs and targets."""

    def __init__(self, return_ids: bool = False) -> None:
        self.return_ids = return_ids

    def __call__(self, batch: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        # Sort batch by spec length (descending)
        _, sorted_idx = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), descending=True
        )
        batch = [batch[i] for i in sorted_idx]

        batch_size = len(batch)
        spec_dim = batch[0][0].size(0)
        phone_dim = batch[0][2].shape[1]
        max_spec_len = max(x[0].size(1) for x in batch)
        max_wave_len = max(x[1].size(1) for x in batch)
        max_phone_len = max(x[2].size(0) for x in batch)

        spec_padded = torch.zeros(batch_size, spec_dim, max_spec_len)
        wave_padded = torch.zeros(batch_size, 1, max_wave_len)
        phone_padded = torch.zeros(batch_size, max_phone_len, phone_dim)
        pitch_padded = torch.zeros(batch_size, max_phone_len, dtype=torch.long)
        pitchf_padded = torch.zeros(batch_size, max_phone_len)
        spec_lengths = torch.zeros(batch_size, dtype=torch.long)
        wave_lengths = torch.zeros(batch_size, dtype=torch.long)
        phone_lengths = torch.zeros(batch_size, dtype=torch.long)
        sid = torch.zeros(batch_size, dtype=torch.long)

        for i, row in enumerate(batch):
            spec, wave, phone, pitch, pitchf, dv = row
            spec_padded[i, :, : spec.size(1)] = spec
            wave_padded[i, :, : wave.size(1)] = wave
            phone_padded[i, : phone.size(0), :] = phone
            pitch_padded[i, : pitch.size(0)] = pitch
            pitchf_padded[i, : pitchf.size(0)] = pitchf
            spec_lengths[i] = spec.size(1)
            wave_lengths[i] = wave.size(1)
            phone_lengths[i] = phone.size(0)
            sid[i] = dv

        return [
            phone_padded,
            phone_lengths,
            pitch_padded,
            pitchf_padded,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
        ]


class TextAudioLoader(torch.utils.data.Dataset):
    """
    Loads audio and text pairs, normalizes text, converts to sequences of integers,
    and computes spectrograms from audio files.
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
        self.lengths = []

    def get_audio_text_pair(self, entry: tuple[Path, Path, int]) -> tuple:
        file, phone_path, dv = entry
        phone = self.get_labels(phone_path)
        spec, wav = self.get_audio(file)
        dv_tensor = torch.LongTensor([dv])

        len_min = min(phone.size(0), spec.size(-1))
        if phone.size(0) != spec.size(-1):
            spec = spec[:, :len_min]
            wav = wav[:, : len_min * self.hop_length]
            phone = phone[:len_min, :]
        return spec, wav, phone, dv_tensor

    def get_labels(self, phone_path: Path) -> torch.FloatTensor:
        phone = np.load(phone_path)
        phone = np.repeat(phone, 2, axis=0)[: min(phone.shape[0] * 2, 900), :]
        return torch.FloatTensor(phone)

    def get_audio(self, filename: Path) -> tuple[torch.Tensor, torch.Tensor]:
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR"
            )
        audio_norm = audio.unsqueeze(0)
        spec_filename = filename.with_suffix(".spec.pt")
        if spec_filename.exists():
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
    """Zero-pads model inputs and targets."""

    def __init__(self, return_ids: bool = False) -> None:
        self.return_ids = return_ids

    def __call__(self, batch: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        # Sort batch by spec length (descending)
        _, sorted_idx = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), descending=True
        )
        batch = [batch[i] for i in sorted_idx]

        batch_size = len(batch)
        spec_dim = batch[0][0].size(0)
        phone_dim = batch[0][2].shape[1]
        max_spec_len = max(x[0].size(1) for x in batch)
        max_wave_len = max(x[1].size(1) for x in batch)
        max_phone_len = max(x[2].size(0) for x in batch)

        spec_padded = torch.zeros(batch_size, spec_dim, max_spec_len)
        wave_padded = torch.zeros(batch_size, 1, max_wave_len)
        phone_padded = torch.zeros(batch_size, max_phone_len, phone_dim)
        spec_lengths = torch.zeros(batch_size, dtype=torch.long)
        wave_lengths = torch.zeros(batch_size, dtype=torch.long)
        phone_lengths = torch.zeros(batch_size, dtype=torch.long)
        sid = torch.zeros(batch_size, dtype=torch.long)

        for i, row in enumerate(batch):
            spec, wave, phone, dv = row
            spec_padded[i, :, : spec.size(1)] = spec
            wave_padded[i, :, : wave.size(1)] = wave
            phone_padded[i, : phone.size(0), :] = phone
            spec_lengths[i] = spec.size(1)
            wave_lengths[i] = wave.size(1)
            phone_lengths[i] = phone.size(0)
            sid[i] = dv

        return [
            phone_padded,
            phone_lengths,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
        ]


class DistributedBucketSampler(torch.utils.data.DistributedSampler):
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
