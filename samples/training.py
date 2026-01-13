import collections
import datasets
import glob
import io
import os
import kagglehub
import librosa
import matplotlib.pyplot as plt
import numpy as np
import openwakeword
import pandas as pd
import scipy
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from typing import Optional, Union


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path: str, sample_rate: int, enable_mono: bool, ext_name: str = "wav"):
        self.enable_mono = enable_mono
        self.sample_rate = sample_rate
        self.files_path = sorted(glob.glob(os.path.join(dir_path, f"*.{ext_name}")))

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx):
        file_path = self.files_path[idx]
        tensor_data, sr = torchaudio.load(file_path)  # [channels, samples]
        np_data = self.transform(tensor_data.numpy(), sr, self.sample_rate, self.enable_mono)
        return dict(
            audio=dict(
                path=file_path,
                sampling_rate=sr,
                array=torch.from_numpy(np_data),
            )
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


class CV17Dataset:
    def __init__(
        self,
        hf_path: str,
        language: str = "en",
        n_train: int = 100,
        n_test: int = 20,
        sample_rate: int = 16000,
        enable_mono: bool = True,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        self.cache_dir = f"{hf_path}/_cache"
        self.train_dir = f"{hf_path}/common_voice_17/{language}/train"
        self.test_dir = f"{hf_path}/common_voice_17/{language}/test"
        self.dataset_name = "fixie-ai/common_voice_17_0"
        self.language = language
        self.n_train = n_train
        self.n_test = n_test
        self.sample_rate = sample_rate
        self.enable_mono = enable_mono
        self.trainloader = torch.utils.data.DataLoader(
            self.load_dataset("train"),
            batch_size=batch_size,
            collate_fn=self.audio_collate_fn,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.load_dataset("test"),
            batch_size=batch_size,
            collate_fn=self.audio_collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

    def load_dataset(self, split_name: str):
        dataset = AudioDataset(
            dir_path=self.try_download(split_name),
            sample_rate=self.sample_rate,
            enable_mono=True,
        )
        print(f"Load audio files: {len(dataset)} for {split_name}")
        return dataset

    def try_download(self, split_name: str):
        if split_name not in ["train", "test"]:
            raise ValueError(f"Invalid split_name: {split_name}")

        if split_name == "train":
            n_element = self.n_train
            output_dir = self.train_dir
        else:
            n_element = self.n_test
            output_dir = self.test_dir

        os.makedirs(output_dir, exist_ok=True)
        audio_dataset = AudioDataset(
            dir_path=output_dir,
            sample_rate=self.sample_rate,
            enable_mono=True,
        )
        audio_num = len(audio_dataset)
        if audio_num >= n_element:
            print(f"Find audio files: {audio_num} for {split_name}")
            return output_dir

        dataset = datasets.load_dataset(
            self.dataset_name,
            self.language,
            split=split_name,
            streaming=True,
            cache_dir=self.cache_dir,
        ).take(n_element)
        for i, element in enumerate(dataset):
            audio_path = element['audio']['path']
            if not audio_path:
                audio_path = os.path.basename(element['path'])
            audio_data = element['audio']['array']
            audio_sr = element['audio']['sampling_rate']
            sentence = element['sentence']
            print(f"Index[{i}] {audio_path=} {audio_data.shape=} {audio_sr=} {sentence=}")

            audio_base, _ = os.path.splitext(audio_path)
            save_path = f"{output_dir}/cv17_{i}_{audio_base}.wav"
            if not os.path.exists(save_path):
                audio_target = AudioDataset.transform(
                    audio_data, audio_sr, self.sample_rate, self.enable_mono)
                sf.write(save_path, audio_target, self.sample_rate)
        print(f"Download audio files: {n_element} for {split_name}")
        return output_dir

    @staticmethod
    def audio_collate_fn(batch):
        paths = [b["audio"]["path"] for b in batch]
        waveforms = [b["audio"]["array"] for b in batch]
        sample_rates = [b["audio"]["sampling_rate"] for b in batch]

        # [T] -> [1, T]
        waveforms = [
            w if w.ndim == 2 else w.unsqueeze(0)
            for w in waveforms
        ]

        lengths = torch.tensor([w.shape[-1] for w in waveforms])

        max_len = max(w.shape[-1] for w in waveforms)
        padded = torch.zeros(len(waveforms), 1, max_len)
        for i, w in enumerate(waveforms):
            padded[i, :, : w.shape[-1]] = w

        return dict(
            paths=paths,
            sampling_rates=sample_rates,
            lengths=lengths,
            waveforms=padded, # [B, 1, T]
        )


if __name__ == "__main__":
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hf_dir = f"{work_dir}/data/huggingface"
    cv17 = CV17Dataset(hf_path=hf_dir, n_train=5, n_test=2)
    cv17.download(target_sr=16000, enable_mono=True)


#final_ds = Dataset.from_list(list(small_ds))
#final_ds.save_to_disk("./cv_100_samples")
