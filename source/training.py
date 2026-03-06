import collections
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openwakeword
import openwakeword.data
import random
import soundfile as sf
import torch
from tqdm import tqdm
from typing import List, Tuple
from functools import lru_cache
from .dataset import AudioDataset
from .utils import AudioFile
from .utils import AudioConvert


class OWWBaseFeature:
    @staticmethod
    def trim_mmap(mmap_path: str):
        """
        Trims blank rows from the end of a mmaped numpy array by creates new mmap array without the blank rows.
        Note that a copy is created and disk usage will briefly double as the function runs.
        """
        mmap_file1 = np.load(mmap_path, mmap_mode='r')
        i = -1
        while np.all(mmap_file1[i, :, :] == 0):
            i -= 1

        N_new = mmap_file1.shape[0] + i + 1

        # Create new mmap_file and copy over data in batches
        output_file2 = mmap_path.strip(".npy") + "2.npy"
        mmap_file2 = np.lib.format.open_memmap(
            output_file2, mode='w+', dtype=np.float32,
            shape=(N_new, mmap_file1.shape[1], mmap_file1.shape[2]),
        )

        for i in tqdm(range(0, mmap_file1.shape[0], 1024), total=mmap_file1.shape[0]//1024, desc="Trimming empty rows"):
            if i + 1024 > N_new:
                mmap_file2[i:N_new] = mmap_file1[i:N_new].copy()
                mmap_file2.flush()
            else:
                mmap_file2[i:i+1024] = mmap_file1[i:i+1024].copy()
                mmap_file2.flush()

        os.remove(mmap_path)
        os.rename(output_file2, mmap_path)


class OWWNegativeFeature(OWWBaseFeature):
    def __init__(
        self,
        feature_path: str,
        *,
        dataset: AudioDataset,
        window_sec: int,
        batch_size: int,
    ):
        self.feature_path = feature_path
        self.dataset = dataset
        self.window_sec = window_sec  # the desired window size (in seconds) for the trained openWakeWord model
        self.batch_size = batch_size  # number of files to load, compute features, and write to mmap at a time
        self.total_N = int(sum(dataset.durations)//window_sec)  # maximum number of rows in mmap file
        self.F = openwakeword.utils.AudioFeatures()
        self.embedding_shape = self.F.get_embedding_shape(window_sec)
        self.embedding_T = self.embedding_shape[0]
        self.embedding_F = self.embedding_shape[1]
        self.f_mem = None
        print(f"Negative: {self.total_N=} {window_sec=} {batch_size=} {self.embedding_T=} {self.embedding_F=}")

    def __enter__(self):
        # Process files by batch and save to Numpy memory mapped file so that
        # an array larger than the available system memory can be created
        os.makedirs(os.path.dirname(self.feature_path), exist_ok=True)
        self.f_mem = np.lib.format.open_memmap(
            self.feature_path,
            mode='w+',
            dtype=np.float32,
            shape=(self.total_N, self.embedding_T, self.embedding_F),
        )
        return self

    def __exit__(self, *args):
        # Trip empty rows from the mmapped array
        self.trim_mmap(self.feature_path)

    def write(self):
        row_counter = 0
        for i in tqdm(np.arange(0, len(self.dataset), self.batch_size)):
            # Load data in batches and shape into rectangular array
            batch_items = self.dataset.get_batch(start=i, size=self.batch_size)
            batch_data = [(j["audio"]["array"]*32767).astype(np.int16) for j in batch_items]
            clip_size = self.dataset.sample_rate * self.window_sec
            clip_data = self._stack_clips(batch_data, clip_size=clip_size).astype(np.int16)

            # Compute features (increase ncpu argument for faster processing)
            features = self.F.embed_clips(x=clip_data, batch_size=1024, ncpu=8)
            print(
                f"Negative[{i}>{row_counter}]: "
                f"{len(batch_data)=} {batch_data[0].shape=} "
                f"{clip_data.shape=} {features.shape=}"
            )

            # Save computed features to mmap array file (stopping once the desired size is reached)
            if row_counter + features.shape[0] > self.total_N:
                self.f_mem[row_counter:min(row_counter+features.shape[0], self.total_N), :, :] = features[0:self.total_N - row_counter, :, :]
                self.f_mem.flush()
                break
            else:
                self.f_mem[row_counter:row_counter+features.shape[0], :, :] = features
                row_counter += features.shape[0]
                self.f_mem.flush()

    def _stack_clips(self, audio_data: List[np.ndarray], clip_size: int) -> np.ndarray:
        """
        Takes an input list of 1D arrays (of different lengths), concatenates them together,
        and then extracts clips of a uniform size by dividing the combined array.
        """
        result = []
        combined = np.concatenate(audio_data)
        for start in range(0, len(combined), clip_size):
            chunk = combined[start:start+clip_size]
            if len(chunk) < clip_size:
                chunk = np.pad(chunk, (0, clip_size - len(chunk)))
            result.append(chunk)
        return np.array(result)


class OWWPositiveFeature(OWWBaseFeature):
    def __init__(
        self,
        feature_path: str,
        *,
        positive_dataset: AudioDataset,
        negative_dataset: AudioDataset,
        window_sec: int,
        batch_size: int,
    ):
        self.feature_path = feature_path
        self.positive_dataset = positive_dataset  # positive example
        self.negative_dataset = negative_dataset  # negative example
        self.window_sec = window_sec  # the desired window size (in seconds) for the trained openWakeWord model
        self.batch_size = batch_size  # number of files to load, compute features, and write to mmap at a time
        self.total_N = len(positive_dataset)  # maximum number of rows in mmap file
        self.F = openwakeword.utils.AudioFeatures()
        self.embedding_shape = self.F.get_embedding_shape(window_sec)
        self.embedding_T = self.embedding_shape[0]
        self.embedding_F = self.embedding_shape[1]
        self.f_mem = None
        print(f"Positive: {self.total_N=} {window_sec=} {batch_size=} {self.embedding_T=} {self.embedding_F=}")

    def __enter__(self):
        # Process files by batch and save to Numpy memory mapped file so that
        # an array larger than the available system memory can be created
        os.makedirs(os.path.dirname(self.feature_path), exist_ok=True)
        self.f_mem = np.lib.format.open_memmap(
            self.feature_path,
            mode='w+',
            dtype=np.float32,
            shape=(self.total_N, self.embedding_T, self.embedding_F),
        )
        return self

    def __exit__(self, *args):
        # Trip empty rows from the mmapped array
        self.trim_mmap(self.feature_path)

    def write(self):
        # Define starting point for each positive clip based on its length, so that each one ends 
        # between 0-200 ms from the end of the total window size chosen for the model.
        # This results in the model being most confident in the prediction right after the
        # end of the wakeword in the audio stream, reducing latency in operation.

        # Get start and end positions for the positive audio in the full window
        sr = self.positive_dataset.sample_rate
        durations = self.positive_dataset.durations
        jitters = (np.random.uniform(0, 0.2, self.total_N)*sr).astype(np.int32)
        window_size = int(sr * self.window_sec)
        starts = [window_size - (int(np.ceil(i*sr))+j) for i,j in zip(durations, jitters)]
        ends = [int(i*sr) + j for i, j in zip(durations, starts)]

        # Create generator to mix the positive audio with background audio
        mixing_generator = self._mix_clips_batch(
            foreground_clips = self.positive_dataset.file_paths,
            background_clips = self.negative_dataset.file_paths,
            combined_size = window_size,
            batch_size = self.batch_size,
            snr_low = 5,
            snr_high = 15,
            start_index = starts,
            volume_augmentation=True, # randomly scale the volume of the audio after mixing
        )

        row_counter = 0
        for i, (clips_batch, labels_batch, background_batch) in enumerate(
            tqdm(mixing_generator, total=self.total_N//self.batch_size),
        ):
            background_shape = background_batch.shape if background_batch else None

            # Compute audio features
            features = self.F.embed_clips(clips_batch, batch_size=256)
            print(
                f"Positive[{i}>{row_counter}]: "
                f"{clips_batch.shape=} {labels_batch.shape=} {background_shape=} {features.shape=}"
            )

            # Save computed features
            self.f_mem[row_counter:row_counter+features.shape[0], :, :] = features
            row_counter += features.shape[0]
            self.f_mem.flush()

            if row_counter >= self.total_N:
                break

    def _mix_clips_batch(
        self,
        foreground_clips: List[str],
        background_clips: List[str],
        combined_size: int,
        labels: List[int] = [],
        batch_size: int = 32,
        snr_low: float = 0,
        snr_high: float = 0,
        start_index: List[int] = [],
        foreground_durations: List[float] = [],
        foreground_truncate_strategy: str = "random",
        rirs: List[str] = [],
        rir_probability: int = 1,
        volume_augmentation: bool = True,
        generated_noise_augmentation: float = 0.0,
        shuffle: bool = True,
        return_sequence_labels: bool = False,
        return_background_clips: bool = False,
        return_background_clips_delay: Tuple[int, int] = (0, 0),
        seed: int = 0
    ):
        """
        Mixes foreground and background clips at a random SNR level in batches.

        References: https://pytorch.org/audio/main/tutorials/audio_data_augmentation_tutorial.html and
        https://speechbrain.readthedocs.io/en/latest/API/speechbrain.processing.speech_augmentation.html#speechbrain.processing.speech_augmentation.AddNoise

        Returns:
            generator: Returns a generator that yields batches of mixed foreground/background audio, labels, and the
                    background segments used for each audio clip (or None is the `return_backgroun_clips` argument is False)
        """
        # Set random seed, if needed
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        # Check and Set start indices, if needed
        if not start_index:
            start_index = [0]*batch_size
        else:
            if min(start_index) < 0:
                raise ValueError("Error! At least one value of the `start_index` argument is <0. Check your inputs.")

        # Make dummy labels
        if not labels:
            labels = [0]*len(foreground_clips)

        if shuffle:
            p = np.random.permutation(len(foreground_clips))
            foreground_clips = np.array(foreground_clips)[p].tolist()
            start_index = np.array(start_index)[p].tolist()
            labels = np.array(labels)[p].tolist()
            if foreground_durations:
                foreground_durations = np.array(foreground_durations)[p].tolist()

        for i in range(0, len(foreground_clips), batch_size):
            # Load foreground clips/start indices and truncate as needed
            sr = 16000
            start_index_batch = start_index[i:i+batch_size]
            foreground_clips_batch = [read_audio(j) for j in foreground_clips[i:i+batch_size]]
            foreground_clips_batch = [j[0] if len(j.shape) > 1 else j for j in foreground_clips_batch]
            if foreground_durations:
                foreground_clips_batch = [truncate_clip(j, int(k*sr), foreground_truncate_strategy)
                                        for j, k in zip(foreground_clips_batch, foreground_durations[i:i+batch_size])]
            labels_batch = np.array(labels[i:i+batch_size])

            # Load background clips and pad/truncate as needed
            background_clips_batch = [read_audio(j) for j in random.sample(background_clips, batch_size)]
            background_clips_batch = [j[0] if len(j.shape) > 1 else j for j in background_clips_batch]
            background_clips_batch_delayed = []
            delay = np.random.randint(return_background_clips_delay[0], return_background_clips_delay[1] + 1)
            for ndx, background_clip in enumerate(background_clips_batch):
                if background_clip.shape[0] < (combined_size + delay):
                    repeated = background_clip.repeat(
                        np.ceil((combined_size + delay)/background_clip.shape[0]).astype(np.int32)
                    )
                    background_clips_batch[ndx] = repeated[0:combined_size]
                    background_clips_batch_delayed.append(repeated[0+delay:combined_size + delay].clone())
                elif background_clip.shape[0] > (combined_size + delay):
                    r = np.random.randint(0, max(1, background_clip.shape[0] - combined_size - delay))
                    background_clips_batch[ndx] = background_clip[r:r + combined_size]
                    background_clips_batch_delayed.append(background_clip[r+delay:r + combined_size + delay].clone())

            # Mix clips at snr levels
            snrs_db = np.random.uniform(snr_low, snr_high, batch_size)
            mixed_clips = []
            sequence_labels = []
            for fg, bg, snr, start in zip(foreground_clips_batch, background_clips_batch,
                                        snrs_db, start_index_batch):
                if bg.shape[0] != combined_size:
                    raise ValueError(bg.shape)
                mixed_clip = mix_clip(fg, bg, snr, start)
                sequence_labels.append(get_frame_labels(combined_size, start, start+fg.shape[0]))

                if np.random.random() < generated_noise_augmentation:
                    noise_color = ["white", "pink", "blue", "brown", "violet"]
                    noise_clip = acoustics.generator.noise(combined_size, color=np.random.choice(noise_color))
                    noise_clip = torch.from_numpy(noise_clip/noise_clip.max())
                    mixed_clip = mix_clip(mixed_clip, noise_clip, np.random.choice(snrs_db), 0)

                mixed_clips.append(mixed_clip)

            mixed_clips_batch = torch.vstack(mixed_clips)
            sequence_labels_batch = torch.from_numpy(np.vstack(sequence_labels))

            # Apply reverberation to the batch (from a single RIR file)
            if rirs:
                if np.random.random() <= rir_probability:
                    rir_waveform, sr = torchaudio.load(random.choice(rirs))
                    if rir_waveform.shape[0] > 1:
                        rir_waveform = rir_waveform[random.randint(0, rir_waveform.shape[0]-1), :]
                    mixed_clips_batch = reverberate(mixed_clips_batch, rir_waveform, rescale_amp="avg")

            # Apply volume augmentation
            if volume_augmentation:
                volume_levels = np.random.uniform(0.02, 1.0, mixed_clips_batch.shape[0])
                mixed_clips_batch = (volume_levels/mixed_clips_batch.max(dim=1)[0])[..., None]*mixed_clips_batch
            else:
                # Normalize clips only if max value is outside of [-1, 1]
                abs_max, _ = torch.max(
                    torch.abs(mixed_clips_batch), dim=1, keepdim=True
                )
                mixed_clips_batch = mixed_clips_batch / abs_max.clamp(min=1.0)

            # Convert to 16-bit PCM audio
            mixed_clips_batch = (mixed_clips_batch.numpy()*32767).astype(np.int16)

            # Remove any clips that are silent (happens rarely when mixing/reverberating)
            error_index = torch.from_numpy(np.where(mixed_clips_batch.max(axis=1) != 0)[0])
            mixed_clips_batch = mixed_clips_batch[error_index]
            labels_batch = labels_batch[error_index]
            sequence_labels_batch = sequence_labels_batch[error_index]

            if not return_background_clips:
                yield mixed_clips_batch, labels_batch if not return_sequence_labels else sequence_labels_batch, None
            else:
                background_clips_batch_delayed = (torch.vstack(background_clips_batch_delayed).numpy()
                                                * 32767).astype(np.int16)[error_index]
                yield (mixed_clips_batch,
                    labels_batch if not return_sequence_labels else sequence_labels_batch,
                    background_clips_batch_delayed)


class OWWDataset:
    def __init__(self, window_sec: int, sample_rate: int, enable_mono: bool):
        self.sample_rate = sample_rate
        self.enable_mono = enable_mono
        self.window_sec = window_sec
        self.positive_features = None
        self.negative_features = None
        self.train_batch_size = None
        self.trainloader = None

    @property
    def total_N(self) -> int:
        if self.trainloader is None or self.train_batch_size is None:
            raise RuntimeError("Invalid dataset")
        return len(self.trainloader) * self.train_batch_size

    @property
    def embedding_T(self) -> int:
        if self.trainloader is None:
            raise RuntimeError("Invalid dataset")
        return self.positive_features.shape[1]
    
    @property
    def embedding_F(self) -> int:
        if self.trainloader is None:
            raise RuntimeError("Invalid dataset")
        return self.positive_features.shape[2]

    def save_negative_features(self, feature_path: str, *, negative_dirs: List[str]):
        print(f"Negative: {negative_dirs=}")
        dataset = AudioDataset(
            target_dirs=negative_dirs,
            sample_rate=self.sample_rate,
            enable_mono=self.enable_mono,
        )
        print(f"Negative: {len(dataset)=} with ~{sum(dataset.durations)} sec, {feature_path=}")
        with OWWNegativeFeature(
            feature_path,
            dataset=dataset,
            window_sec=self.window_sec,
            batch_size=64,
        ) as oww_feature:
            oww_feature.write()

    def save_positive_features(self, feature_path: str, *, positive_dirs: List[str], negative_dirs: List[str]):
        print(f"Positive: {positive_dirs=} {negative_dirs=}")
        negative_dataset = AudioDataset(
            target_dirs=negative_dirs,
            sample_rate=self.sample_rate,
            enable_mono=self.enable_mono,
        )
        print(f"Positive: {len(negative_dataset)=} with ~{sum(negative_dataset.durations)} sec.")
        positive_dataset = AudioDataset(
            target_dirs=positive_dirs,
            sample_rate=self.sample_rate,
            enable_mono=self.enable_mono,
        )
        print(f"Positive: {len(positive_dataset)=} with ~{sum(positive_dataset.durations)} sec, {feature_path=}")
        with OWWPositiveFeature(
            feature_path,
            positive_dataset=positive_dataset,
            negative_dataset=negative_dataset,
            window_sec=self.window_sec,
            batch_size=8,
        ) as oww_feature:
            oww_feature.write()

    def load_features(
        self,
        positive_feature_path: str,
        negative_feature_path: str,
        batch_size: int,
    ):
        # Load the data prepared in previous steps (it's small enough to load entirely in memory)
        positive_features = np.load(positive_feature_path) # [N_pos, T, F] = (N, 28, 96)
        negative_features = np.load(negative_feature_path) # [N_neg, T, F] = (N, 28, 96)
        if positive_features.shape[1] != negative_features.shape[1]:
            raise RuntimeError(
                f"Invalid embedding timestep: "
                f"positive={positive_features.shape[1]} "
                f"negative={negative_features.shape[1]}"
            )
        if positive_features.shape[2] != negative_features.shape[2]:
            raise RuntimeError(
                f"Invalid embedding feature: "
                f"positive={positive_features.shape[2]} "
                f"negative={negative_features.shape[2]}"
            )

        print(f"Load features positive={positive_features.shape} negative={negative_features.shape}")
        self.positive_features = positive_features
        self.negative_features = negative_features
        self.train_batch_size = batch_size

        X = np.vstack((negative_features, positive_features)).astype(np.float32)
        y = np.concatenate([
            np.zeros(len(negative_features)), # negative -> 0
            np.ones(len(positive_features)),  # positive -> 1
        ]).astype(np.float32)[:, None]
        self.trainloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X),
                torch.from_numpy(y),
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

    def show(self):
        if self.trainloader is None:
            raise RuntimeError("Invalid dataset")
        print(
            f"Features Dataset:"
            f"\n  {len(self.trainloader)=}"
            f"\n  {self.train_batch_size=}"
            f"\n  {self.total_N=}"
            f"\n  {self.embedding_T=}"
            f"\n  {self.embedding_F=}"
        )
        for x_batch, y_batch in self.trainloader:
            print(f"  {x_batch.shape=}") # [B, T, F] = (512, 28, 96)
            print(f"  {y_batch.shape=}") # [B, 1] = (512, 1)
            break


class OWWNetwork(torch.nn.Module):
    def __init__(self, *, embedding_timestep: int, embedding_feature: int, layer_dim: int):
        super().__init__()
        self.flatten = torch.nn.Flatten() # the input is flattened
        self.fc1 = torch.nn.Linear(embedding_timestep * embedding_feature, layer_dim)
        self.ln1 = torch.nn.LayerNorm(layer_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(layer_dim, layer_dim)
        self.ln2 = torch.nn.LayerNorm(layer_dim)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(layer_dim, 1)

    def forward(self, x):
        # x shape: [B, T, F]
        # Remove the last Sigmoid, BCEWithLogitsLoss will handle it internally
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class OWWModel:
    def __init__(
        self,
        dataset: OWWDataset,
        *,
        layer_dim: int,
        learn_rate: float,
        positive_weight: float = 0.1,
    ):
        self.dataset = dataset
        self.network = OWWNetwork(
            embedding_timestep=dataset.embedding_T,
            embedding_feature=dataset.embedding_F,
            layer_dim=layer_dim,
        ).to(self.device)
        self.positive_weight = positive_weight
        self.criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.positive_weight], device=self.device),
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learn_rate)
        self.metadata = collections.defaultdict(list)
        print(f"{self.network=}")

    @property
    @lru_cache(maxsize=1)
    def device(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Device: use {device}")
        return device

    def train(self, num_epochs: int):
        print(f"Train: {num_epochs=}")
        self.metadata.clear()
        self.network.train()
        for i in tqdm(range(num_epochs), total=num_epochs):
            for batch in self.dataset.trainloader:
                # Get data for batch
                x, y = batch[0].to(self.device), batch[1].to(self.device)

                # Give a higher weight to the negative sample to prevent false positives
                # As you have more data (both positive and negative), this is less important
                weights = torch.ones_like(y, device=y.device)
                weights[y == 1] = self.positive_weight

                # Zero gradients
                self.optimizer.zero_grad()
                
                # Run forward pass
                logits = self.network(x)
                
                # Update model parameters
                loss_per_sample = self.criterion(logits, y)
                loss = (loss_per_sample * weights).mean()
                loss.backward()
                self.optimizer.step()

                # Log metrics
                with torch.no_grad():
                    probs = torch.sigmoid(logits).flatten()
                    labels = y.flatten()
                    pos_mask = labels == 1
                    if pos_mask.any():
                        tp = (probs[pos_mask] >= 0.5).sum() # True Positive
                        fn = (probs[pos_mask] < 0.5).sum()  # False Negative
                        recall = (tp / (tp + fn)).item()
                    else:
                        recall = 0.0
                    self.metadata['loss'].append(loss.item())
                    self.metadata['recall'].append(recall)

    @torch.no_grad()
    def predict(self, audio_path: str):
        # Pre-compute audio features using helper function
        F = openwakeword.utils.AudioFeatures()
        audio_data = AudioFile(audio_path).transform(
            sample_rate=self.dataset.sample_rate,
            enable_mono=self.dataset.enable_mono,
            dtype='int16',
        )
        features = F._get_embeddings(audio_data) # [N, F]
        features_N = features.shape[0]

        # Get predictions for each window
        self.network.eval()
        scores = []
        with torch.no_grad():
            for i in tqdm(range(0, features_N - self.dataset.embedding_T)):
                window = features[i:i + self.dataset.embedding_T]       # [T, F]
                window = torch.from_numpy(window).float().unsqueeze(0)  # [1, T, F]
                logits = self.network(window.to(self.device))           # [1, 1]
                prob = torch.sigmoid(logits)       # Add sigmoid when reasoning
                scores.append(float(prob.item()))
                print(f"Predict[{i}]: {prob=} {logits=} {window.shape=}")
        print(f"Predict: {audio_path} {features.shape=} {len(scores)=}")
        return scores

    @torch.no_grad()
    def predict_with_mixmusic(self, audio_path: str, music_path: str):
        output_path = f"{os.path.splitext(audio_path)[0]}_music.wav"
        audio_data, audio_sr = sf.read(audio_path)
        music_data, music_sr = sf.read(music_path)
        mixing_data = AudioConvert.mixing(
            audio_data,
            audio_sr,
            secondary=music_data,
            secondary_sr=music_sr,
            secondary_gain=0.7,
            output_sr=self.dataset.sample_rate,
            start_sec=0.0,
            duration_sec=20,
            enable_mono=self.dataset.enable_mono,
        )
        sf.write(output_path, mixing_data, self.dataset.sample_rate)
        return self.predict(output_path)

    def save_onnx(self, output_path: str):
        self.network.eval()
        torch.set_grad_enabled(False)

        # the 'args' is the shape of a single example
        dummy_input = torch.zeros(
            (1, self.dataset.embedding_T, self.dataset.embedding_F),
            device=self.device,
        )
        torch.onnx.export(
            self.network,
            args=dummy_input,
            f=output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['logits'],
            dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
        )

    def onnx_predict(self, onnx_path: str, audio_path: str):
        # Pre-compute audio features using helper function
        F = openwakeword.utils.AudioFeatures()
        audio_data = AudioFile(audio_path).transform(
            sample_rate=self.dataset.sample_rate,
            enable_mono=self.dataset.enable_mono,
            dtype='int16',
        )
        features = F._get_embeddings(audio_data) # [N, F]
        features_N = features.shape[0]

        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        scores = []
        for i in tqdm(range(0, features_N - self.dataset.embedding_T)):
            window = features[i:i + self.dataset.embedding_T]       # [T, F]
            window_np = (window.astype(np.float32)[None, :, :])     # [1, T, F]
            logits = session.run(None, {input_name: window_np})[0]  # [1, 1]
            prob = 1 / (1 + np.exp(-logits))                        # Add sigmoid when reasoning
            scores.append(float(prob.item()))
            print(f"ONNXPredict[{i}]: {prob=} {logits=} {window.shape=} {window_np.shape=}")
        print(f"ONNXPredict: {audio_path} {features.shape=} {len(scores)=}")
        return scores

    def plot_metrics(self, save_path: str):
        matplotlib.use("Agg")
        plt.figure()
        plt.plot(self.metadata['loss'], label="loss")
        plt.plot(self.metadata['recall'], label="recall")
        plt.legend()
        plt.ylim(0,1)
        plt.savefig(save_path)
        plt.close()

    def plot_scores(self, save_path: str, scores: list[float]):
        matplotlib.use("Agg")
        plt.figure()
        plt.plot(scores)
        plt.ylim(0,1)
        plt.savefig(save_path)
        plt.close()
