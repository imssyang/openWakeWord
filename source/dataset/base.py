import os
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from typing import List
from ..utils import AudioConvert


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
        np_data = AudioConvert.transform(
            tensor_data.numpy(),
            orig_sr=sr,
            target_sr=self.sample_rate,
            enable_mono=self.enable_mono)
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
        file_paths = []
        durations = []
        for target_dir in target_dirs:
            sizes = []
            dir_paths = [str(i) for i in Path(target_dir).glob(f"*.{ext_name}")]
            file_paths.extend(dir_paths)
            sizes.extend([os.path.getsize(i) for i in dir_paths])
            durations.extend([self._get_duration(i) for i in tqdm(dir_paths)])

        filtered = [(i, j) for i, j in zip(file_paths, durations) if j >= self.min_length_secs and j <= self.max_length_secs]
        return [i[0] for i in filtered], [i[1] for i in filtered]

    def _get_duration(self, audio_path):
        try:
            metadata = torchaudio.info(audio_path)
        except RuntimeError:  # skip cases where file metadata can't be read
            return 0
        return metadata.num_frames/metadata.sample_rate
