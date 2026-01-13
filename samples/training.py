import collections
import csv
import datasets
import glob
import io
import os
import kagglehub
import librosa
import matplotlib.pyplot as plt
import numpy as np
import openwakeword
import openwakeword.data
import pandas as pd
import pathlib
import scipy
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from typing import Optional, List


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
        file_path = self.file_paths[idx]
        tensor_data, sr = torchaudio.load(file_path)  # [channels, samples]
        np_data = self.transform(tensor_data.numpy(), sr, self.sample_rate, self.enable_mono)
        return dict(
            audio=dict(
                path=file_path,
                duration=self.durations[idx],
                sampling_rate=sr,
                array=torch.from_numpy(np_data),
            )
        )

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


class CV17Dataset:
    def __init__(
        self,
        hf_path: str,
        batch_size: int,
        n_train: int,
        n_test: int,
        num_workers: int = 0,
        language: str = "en",
        sample_rate: int = 16000,
        enable_mono: bool = True,
    ):
        self.cache_dir = f"{hf_path}/_cache"
        self.dataset_dir = f"{hf_path}/common_voice_17/{language}"
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
        audio_dataset = AudioDataset(
            target_dirs=[output_dir],
            sample_rate=self.sample_rate,
            enable_mono=True,
        )
        audio_num = len(audio_dataset)
        if audio_num >= n_element:
            print(f"Find audio files: {audio_num} for {split_name}")
            return output_dir

        csv_path = f"{self.dataset_dir}/{split_name}_meta.csv"
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
                audio_target = AudioDataset.transform(
                    audio_data, audio_sr, self.sample_rate, self.enable_mono)
                sf.write(save_path, audio_target, self.sample_rate)
                self.append_csv(csv_path, i, save_name, sentence)
        print(f"Download audio files: {n_element} for {split_name}")
        return output_dir

    @staticmethod
    def append_csv(csv_path: str, index: int, filename: str, text: str):
        fieldnames = [
            "index",
            "filename",
            "text",
        ]
        with open(pathlib.Path(csv_path), "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(dict(
                index=index,
                filename=filename,
                text=text,
            ))

    @staticmethod
    def audio_collate_fn(batch):
        paths = [b["audio"]["path"] for b in batch]
        durations = [b["audio"]["duration"] for b in batch]
        sample_rates = [b["audio"]["sampling_rate"] for b in batch]
        waveforms = [b["audio"]["array"] for b in batch]

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


class OWWFeature:
    def __init__(
        self,
        dataset: AudioDataset,
        feature_path: str,
        window_sec: int,
        batch_size: int,
    ):
        self.dataset = dataset
        self.feature_path = feature_path
        self.window_sec = window_sec  # the desired window size (in seconds) for the trained openWakeWord model
        self.batch_size = batch_size  # number of files to load, compute features, and write to mmap at a time
        self.F = openwakeword.utils.AudioFeatures()
        self.shape = self.F.get_embedding_shape(window_sec)
        self.total_N = int(sum(self.dataset.durations)//self.window_sec)
        self.embedding_T = self.shape[0]
        self.embedding_D = self.shape[1]
        self.f_mem = None
        print(f"{feature_path=} {window_sec=} {total_N=} {embedding_T=} {embedding_D=}")

    def __enter__(self):
        # Process files by batch and save to Numpy memory mapped file so that
        # an array larger than the available system memory can be created
        self.f_mem = np.lib.format.open_memmap(
            self.feature_path,
            mode='w+',
            dtype=np.float32,
            shape=(self.total_N, self.embedding_T, self.embedding_D),
        )
        return self

    def __exit__(self, *args):
        # Trip empty rows from the mmapped array
        openwakeword.data.trim_mmap(self.feature_path)

    def write(self):
        row_counter = 0
        for i in tqdm(np.arange(0, len(self.dataset), self.batch_size)):
            # Load data in batches and shape into rectangular array
            wav_data = [(j["array"]*32767).astype(np.int16) for j in self.dataset[i:i+self.batch_size]["audio"]]
            print(f"file_wav: {len(wav_data)=} {wav_data[0].shape=}")
            wav_data = openwakeword.data.stack_clips(wav_data, clip_size=16000*clip_size).astype(np.int16)
            print(f"stack_clips: {len(wav_data)=} {wav_data.shape=}")
            
            # Compute features (increase ncpu argument for faster processing)
            features = self.F.embed_clips(x=wav_data, batch_size=1024, ncpu=8)
            print(f"features: {len(features)=} {features.shape=}")
            
            # Save computed features to mmap array file (stopping once the desired size is reached)
            if row_counter + features.shape[0] > self.total_N:
                self.f_mem[row_counter:min(row_counter+features.shape[0], self.total_N), :, :] = features[0:self.total_N - row_counter, :, :]
                self.f_mem.flush()
                break
            else:
                self.f_mem[row_counter:row_counter+features.shape[0], :, :] = features
                row_counter += features.shape[0]
                self.f_mem.flush()


class OWWDataset:
    def __init__(
        self,
        negative_dirs: List[str],
        positive_dirs: List[str],
        negative_feature_path: str,
        positive_feature_path: str,
        batch_size: int = 512,
        sample_rate: int = 16000,
        enable_mono: bool = True,
    ):
        self.sample_rate = sample_rate
        self.enable_mono = enable_mono
        self.negative_feature_path = negative_feature_path
        self.positive_feature_path = positive_feature_path


        # Load the data prepared in previous steps (it's small enough to load entirely in memory)
        # negative_features: [N_neg, T, F] = (23, 28, 96) 
        # positive_features: [N_pos, T, F] = (3203, 28, 96)
        negative_features = np.load(negative_feat_path)
        positive_features = np.load(positive_feat_path)
        print(f"negative_features len={len(negative_features)} shape={negative_features.shape}")
        print(f"positive_features len={len(positive_features)} shape={positive_features.shape}")

        # X_shape: [N_neg + N_pos, T, F] = (3226, 28, 96)
        # y_shape: [N_neg + N_pos, 1] = (3226, 1)
        X = np.vstack((negative_features, positive_features))
        y = np.array(
            [0]*len(negative_features) +   # negative -> 0
            [1]*len(positive_features)     # positive -> 1
        ).astype(np.float32)[..., None]
        print(f"X len={len(X)} shape={X.shape}")
        print(f"y len={len(y)} shape={y.shape}")

        # x_batch: [B, T, F] = (512, 28, 96)
        # y_batch: [B, 1] = (512, 1)
        self.trainloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X.astype(np.float32)),
                torch.from_numpy(y.astype(np.float32)),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        for x_batch, y_batch in self.trainloader:
            print(f"x_batch len={len(x_batch)} shape={x_batch.shape}")
            print(f"y_batch len={len(y_batch)} shape={y_batch.shape}")
            self.input_timesteps = x_batch.shape[1]
            self.input_feature = x_batch.shape[2]
            print(f"y len={len(y)} shape={y.shape} {self.input_timesteps=} {self.input_feature=}")
            break

    def negative_features(self, target_dirs: List[str], feature_path: str):
        dataset = AudioDataset(
            target_dirs=target_dirs,
            sample_rate=self.sample_rate,
            enable_mono=self.enable_mono,
        )
        print(f"negative_clips: {len(dataset)=} with ~{sum(dataset.durations)} sec")

        with OWWFeature(dataset, feature_path, window_sec=3, batch_size=64) as oww_feature:
            oww_feature.write()





class OWWNetwork(torch.nn.Module):
    def __init__(self, input_timesteps: int, input_feature: int, layer_dim: int = 32):
        super().__init__()
        self.flatten = torch.nn.Flatten() # the input is flattened
        self.fc1 = torch.nn.Linear(input_timesteps * input_feature, layer_dim)
        self.ln1 = torch.nn.LayerNorm(layer_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(layer_dim, layer_dim)
        self.ln2 = torch.nn.LayerNorm(layer_dim)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(layer_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, T, F]
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class OWWModel:
    def __init__(self, input_timesteps, input_feature):
        self.network = OWWNetwork(input_timesteps, input_feature)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

    def train(self, data_loader: torch.utils.data.DataLoader, num_epochs: int = 10):
        # Define training loop, metrics, and logging
        history = collections.defaultdict(list)
        for i in tqdm(range(num_epochs), total=num_epochs):
            for batch in data_loader:
                # Get data for batch
                x, y = batch[0], batch[1]

                # Get weights for classes, and assign 10x higher weight to negative class
                # to help the model learn to not have too many false-positives
                # As you have more data (both positive and negative), this is less important
                weights = torch.ones(y.shape[0])
                weights[y.flatten() == 1] = 0.1

                # Zero gradients
                self.optimizer.zero_grad()
                
                # Run forward pass
                predictions = self.network(x)
                
                # Update model parameters
                loss_per_sample = self.criterion(predictions, y)
                loss = (loss_per_sample * weights).mean()
                loss.backward()
                self.optimizer.step()

                # Log metrics
                history['loss'].append(float(loss.detach().numpy()))
                tp = sum(predictions.flatten()[y.flatten() == 1] >= 0.5)
                fn = sum(predictions.flatten()[y.flatten() == 1] < 0.5)
                history['recall'].append(float(tp/(tp+fn).detach().numpy()))

        # Plot training metrics
        plt.figure()
        plt.plot(history['loss'], label="loss")
        plt.plot(history['recall'], label="recall")
        plt.legend()
        plt.ylim(0,1)


if __name__ == "__main__":
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hf_dir = f"{work_dir}/data/huggingface"
    cv17 = CV17Dataset(hf_path=hf_dir, n_train=5, n_test=2)

    oww_dataset = OWWDataset(
        negative_feat_path="data/huggingface/common_voice_17/en/train_features.npy",
        positive_feat_path="data/turn_on_the_office_lights/positive_features.npy",
        batch_size=512,
    )

    oww_model = OWWModel(
        oww_dataset.input_timesteps, oww_dataset.input_feature
    )
    oww_model.train(oww_dataset.trainloader, num_epochs=10)
