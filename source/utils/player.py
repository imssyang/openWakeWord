from __future__ import annotations
import logging
import threading
import time
from typing import List

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from .convert import AudioConvert


class AudioAttribute:
    """The parent class of public audio attributes, and the subclass can be directly inherited"""

    def __init__(self, *, samplerate: int, channels: int, dtype: str = 'float32'):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = np.dtype(dtype)


class AudioBuffer(AudioAttribute):
    """
    Lock-protected ring buffer for real-time audio.

    Data layout:
        buffer shape = (capacity, channels)
    """

    def __init__(self, capacity: int, **kwargs):
        super().__init__(**kwargs)
        self.capacity = capacity

        # One-time allocation, no memory is allocated during the callback
        self.buffer = np.zeros((capacity, self.channels), dtype=self.dtype)
        self.write_pos = 0
        self.read_pos = 0
        self.lock = threading.Lock()

        # Statistics
        self.available_frame = 0  # The number of valid frames in the current buffer
        self.total_written = 0   # Total write frames
        self.total_read = 0      # Total playback frames
        self.underrun_count = 0  # The number of playback stutters
        self.overflow_count = 0  # The number of written too fast

    def reset(self):
        """Reset ring buffer state and clear all data."""
        with self.lock:
            self.read_pos = 0
            self.write_pos = 0
            self.available_frames = 0
            self.total_written = 0
            self.total_read = 0
            self.underrun_count = 0
            self.overflow_count = 0

            # clear audio data to avoid stale playback
            self.buffer.fill(0)

    def write(self, data: np.ndarray):
        """
        Write audio frames into ring buffer.

        data shape: (N, channels)
        """
        if data.ndim == 1:
            data = data[:, None]

        assert data.shape[1] == self.channels

        with self.lock:
            frames = data.shape[0]
            if frames > self.capacity - self.available_frame:
                # Insufficient buffer zone, discard excess (real-time system must make trade-offs)）
                self.overflow_count += 1
                frames = self.capacity - self.available_frame
                data = data[:frames]
                logging.warning(f"Audio data overflow count {self.overflow_count}")

            first = min(frames, self.capacity - self.write_pos)
            second = frames - first

            self.buffer[self.write_pos:self.write_pos + first] = data[:first]
            if second > 0:
                self.buffer[0:second] = data[first:first + second]

            self.write_pos = (self.write_pos + frames) % self.capacity
            self.available_frame += frames
            self.total_written += frames

    def read(self, frames: int) -> np.ndarray:
        """
        Read exactly `frames` frames from buffer.
        If not enough data, zeros will be returned.
        """
        out = np.zeros((frames, self.channels), dtype=self.dtype)

        with self.lock:
            if self.available_frame < frames:
                self.underrun_count += 1

            read_frames = min(frames, self.available_frame)
            first = min(read_frames, self.capacity - self.read_pos)
            second = read_frames - first

            out[:first] = self.buffer[self.read_pos:self.read_pos + first]
            if second > 0:
                out[first:first + second] = self.buffer[0:second]

            self.read_pos = (self.read_pos + read_frames) % self.capacity
            self.available_frame -= read_frames
            self.total_read += read_frames

        return out

    def drop_oldest(self, frames: int):
        """
        Drop the oldest frames from the buffer.
        Used for low-latency real-time audio (e.g. mic monitoring).

        Args:
            frames: number of frames to drop
        """
        if frames <= 0 or self.available_frame == 0:
            return

        drop = min(frames, self.available_frame)

        # move read pointer forward
        self.read_pos = (self.read_pos + drop) % self.capacity
        self.available_frame -= drop

    @property
    def available_size(self) -> float:
        """Current buffered audio size (latency)."""
        with self.lock:
            return self.available_frame

    @property
    def available_second(self) -> float:
        """Current buffered audio duration (latency)."""
        with self.lock:
            return self.available_frame / self.samplerate

    @property
    def written_second(self) -> float:
        """Total audio duration written to the player since start."""
        with self.lock:
            return self.total_written / self.samplerate

    @property
    def played_second(self) -> float:
        """Total audio duration actually played."""
        with self.lock:
            return self.total_read / self.samplerate


class AudioPlayer(AudioAttribute):
    def __init__(
        self,
        *,
        latency_sec: float = 0.1,
        capacity_sec: float = 3,
        odevice: int | None = None,
         **kwargs,
    ):
        super().__init__(**kwargs)
        self.latency_sec = latency_sec
        self.capacity_sec = capacity_sec
        self.buffer = AudioBuffer(
            capacity=int(capacity_sec * self.samplerate),
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=str(self.dtype),
        )
        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=str(self.dtype),
            device=odevice,
            blocksize=int(latency_sec * self.samplerate),
            callback=self._callback,
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def _callback(self, outdata, frames, time, status):
        if status:
            logging.warning("Stream status:", status)

        data = self.buffer.read(frames)
        if data.shape[1] != self.stream.channels:
            # broadcast to all channels
            data = np.broadcast_to(data, (frames, self.stream.channels))
        outdata[:] = data

    def _add_data(self, data: np.ndarray):
        """
        data shape: (N,) or (N, C)
        """
        if data.ndim == 1:
            data = data[:, None]

        self.buffer.write(data.astype(self.dtype, copy=False))

    def raw_play(self, data: np.ndarray, *, speed: float = 1.0):
        if speed == 1.0:
            data_stretch = data
        else:
            data_stretch = librosa.effects.time_stretch(data.T, rate=speed).T

        self._add_data(data_stretch)
        self.start()

    def start(self):
        if not self.stream.active:
            self.stream.start()

    def stop(self):
        if self.stream.active:
            self.stream.stop()

    def close(self):
        self.stop()
        self.stream.close()

    @staticmethod
    def raw_play_once(data: np.ndarray, samplerate: int, dtype: str = 'float32'):
        data = AudioConvert.convert(data, np.dtype(dtype))
        channels = data.shape[1] if data.ndim > 1 else 1
        with sd.OutputStream(samplerate=samplerate, channels=channels, dtype=dtype) as stream:
            stream.write(data)


class AudioFile(AudioAttribute):
    def __init__(self, audio_path: str, **kwargs):
        dtype = kwargs.get("dtype", "float32")
        data, sr = sf.read(audio_path, dtype=dtype)
        super().__init__(
            samplerate=sr,
            channels=data.shape[1] if data.ndim == 2 else 1,
            dtype=dtype,
            **kwargs,
        )
        self.path = audio_path
        self.data = data

    def transform(self, *, samplerate: int, enable_mono: bool, dtype: str = 'float32'):
        data = AudioConvert.transform(
            self.data,
            orig_sr=self.samplerate,
            target_sr=samplerate,
            enable_mono=enable_mono,
        )
        return AudioConvert.convert(data, np.dtype(dtype))

    def get_chunks(
        self,
        *,
        chunksize: int,
        samplerate: int,
        enable_mono: bool,
        dtype: str = 'float32',
    ) -> List[np.ndarray]:
        chunks = []
        for i in range(0, len(self.data), chunksize):
            chunk = self.data[i:i+chunksize]
            chunk = AudioConvert.convert(chunk, np.dtype(dtype))
            chunks.append(chunk)
        return chunks


class FilePlayer(AudioFile, AudioPlayer):
    def __init__(self, audio_path: str, **kwargs):
        super().__init__(
            audio_path=audio_path,
            latency_sec=0.05,
            capacity_sec=3,            
            **kwargs,
        )

    def play(self, *, speed: float = 1.0):
        """
        Blocking file playback with adjustable speed.

        Args:
            speed: playback speed (>1.0 faster, <1.0 slower)
        """
        self.buffer.reset()

        chunk_sec = 0.05 if speed == 1 else 0.6
        chunk_size = int(chunk_sec * self.samplerate)
        idx = 0
        total_frames = len(self.data)

        while idx < total_frames:
            chunk = self.data[idx:idx+chunk_size]

            # wait if buffer is too full
            while self.buffer.available_size > self.buffer.capacity // 2:
                time.sleep(0.01)

            self.raw_play(chunk, speed=speed)
            idx += chunk_size

        # wait until buffer is consumed
        while self.buffer.available_size > 0:
            time.sleep(0.01)

        self.stop()


class AudioMic(AudioAttribute):
    def __init__(
        self,
        *,
        on_audio: Callable[[np.ndarray], None],
        latency_sec: float = 0.1,
        capacity_sec: float = 3,
        idevice: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.on_audio = on_audio
        self.latency_sec = latency_sec
        self.capacity_sec = capacity_sec
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=str(self.dtype),
            device=idevice,
            blocksize=int(latency_sec * self.samplerate),
            callback=self._callback,
        )

    def _callback(self, indata, frames, time, status):
        if status:
            logging.warning("Input status: %s", status)

        # Guaranteed shape: [frames, channels]
        data = indata.copy()

        if self.on_audio:
            try:
                self.on_audio(data)
            except Exception:
                logging.exception("mic on_audio callback error")
        else:
            if self.buffer is None:
                self.buffer = AudioBuffer(
                    capacity=int(self.capacity_sec * self.samplerate),
                    samplerate=self.samplerate,
                    channels=self.channels,
                    dtype=str(self.dtype),
                )

            # Lose the oldest data when the buffer is full (low-latency monitoring is preferred)
            if self.buffer.available_size + frames > self.buffer.capacity:
                drop = self.buffer.available_size + frames - self.buffer.capacity
                self.buffer.drop_oldest(drop)
            self.buffer.write(data)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        if not self.stream.active:
            self.stream.start()

    def stop(self):
        if self.stream.active:
            self.stream.stop()

    def close(self):
        self.stop()
        self.stream.close()


class MicPlayer(AudioPlayer):
    def __init__(
        self,
        *,
        idevice: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mic = AudioMic(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=str(self.dtype),
            latency_sec=self.latency_sec,
            capacity_sec=self.capacity_sec,
            idevice=idevice,
            on_audio=self._on_audio,
        )

    def _on_audio(self, data: np.ndarray):
        frames = len(data)

        # Lose the oldest data when the buffer is full (low-latency monitoring is preferred)
        if self.buffer.available_size + frames > self.buffer.capacity:
            drop = self.buffer.available_size + frames - self.buffer.capacity
            self.buffer.drop_oldest(drop)

        self.raw_play(data)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        """Start mic monitoring."""
        super().start()
        self.mic.start()

    def stop(self):
        """Stop mic monitoring."""
        self.mic.stop()
        super().stop()

    def close(self):
        """Close mic and player."""
        self.mic.close()
        super().close()
