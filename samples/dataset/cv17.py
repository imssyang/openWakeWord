import csv
import datasets
import os
import pathlib
import soundfile as sf
import torch
from typing import Optional
from .base import AudioDataset
from ..utils import AudioPlayer


class CV17Dataset:
    def __init__(
        self,
        root_dir: str,
        language: str,
        n_train: int,
        n_test: int,
        sample_rate: int,
        enable_mono: bool,
        batch_size: int = 1,
        num_workers: int = 0,
        cache_dir: Optional[str] = None,
    ):
        self.cache_dir = cache_dir
        self.dataset_dir = f"{root_dir}/{language}"
        self.train_dir = f"{self.dataset_dir}/train"
        self.test_dir = f"{self.dataset_dir}/test"
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
            drop_last=True,
            num_workers=num_workers,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.load_dataset("test"),
            batch_size=batch_size,
            collate_fn=self.audio_collate_fn,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
        )

    def load_dataset(self, split_name: str):
        dataset = AudioDataset(
            target_dirs=[self.try_download(split_name)],
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

        csv_path = f"{self.dataset_dir}/{split_name}_meta.csv"
        csv_rows = self.count_csv_rows(csv_path)
        if csv_rows >= n_element:
            print(f"Find audio files: {csv_rows} for {split_name}")
            return output_dir

        if os.path.exists(csv_path):
            os.removedirs(csv_path)

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

            save_name = f"{os.path.splitext(audio_path)[0]}.wav"
            save_path = f"{output_dir}/{save_name}"
            if not os.path.exists(save_path):
                audio_target = AudioPlayer.transform(
                    audio_data, audio_sr, self.sample_rate, self.enable_mono)
                sf.write(save_path, audio_target, self.sample_rate)
                self.append_csv(csv_path, i, save_name, sentence)
        print(f"Download audio files: {n_element} for {split_name}")
        return output_dir

    @staticmethod
    def append_csv(csv_path: str, index: int, filename: str, text: str):
        fieldnames = ["index", "filename", "text"]
        has_header = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(pathlib.Path(csv_path), "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not has_header:
                writer.writeheader()
            writer.writerow(dict(
                index=index,
                filename=filename,
                text=text,
            ))

    @staticmethod
    def count_csv_rows(csv_path: str, has_header: Optional[bool] = None):
        if not os.path.exists(csv_path):
            return 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            if has_header is None:
                sample = f.read(8192)
                f.seek(0)
                has_header = csv.Sniffer().has_header(sample)

            reader = csv.reader(f)
            if has_header:
                next(reader, None)
            return sum(1 for _ in reader)
        raise RuntimeError(f"Read {csv_path=} fail.")

    @staticmethod
    def audio_collate_fn(batch):
        paths = [b["audio"]["path"] for b in batch]
        durations = [b["audio"]["duration"] for b in batch]
        sample_rates = [b["audio"]["sampling_rate"] for b in batch]
        waveforms = [torch.from_numpy(b["audio"]["array"]) for b in batch]

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
            durations=durations,
            sampling_rates=sample_rates,
            lengths=lengths,
            waveforms=padded, # [B, 1, T]
        )

