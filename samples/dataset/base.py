import librosa
import numpy as np
import openwakeword.data
import torch
import torchaudio
from typing import List


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        target_dirs: List[str],
        *,
        sample_rate: int,
        enable_mono: bool,
        min_length_secs: float = 1.0,
        max_length_secs: float = 30.0,
        ext_name: str = "wav",
    ):
        self.enable_mono = enable_mono
        self.sample_rate = sample_rate
        self.min_length_secs = min_length_secs
        self.max_length_secs = max_length_secs
        self.file_paths, self.durations = self._filter_paths(target_dirs, ext_name)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise TypeError("Dataset does not support slicing")
        file_path = self.file_paths[idx]
        tensor_data, sr = torchaudio.load(file_path)  # [channels, samples]
        np_data = self.transform(tensor_data.numpy(), sr, self.sample_rate, self.enable_mono)
        return dict(
            audio=dict(
                path=file_path,
                duration=self.durations[idx],
                sampling_rate=sr,
                array=np_data,
            )
        )

    def get_batch(self, *, start: int, size: int):
        end = min(start + size, len(self))
        return [self[i] for i in range(start, end)]

    def _filter_paths(self, target_dirs: List[str], ext_name: str):
        return openwakeword.data.filter_audio_paths(
            target_dirs,
            min_length_secs=self.min_length_secs,
            max_length_secs=self.max_length_secs,
            duration_method="header",
            glob_filter=f"*.{ext_name}",
        )

    @staticmethod
    def transform(audio_data: np.ndarray, orig_sr: int, target_sr: int, enable_mono: bool) -> np.ndarray:
        if enable_mono:
            audio_data = librosa.to_mono(audio_data)
        if orig_sr != target_sr:
            audio_data = librosa.resample(
                audio_data,
                orig_sr=orig_sr,
                target_sr=target_sr,
            )
        return audio_data

