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


class AudioBuffer:
    """
    Lock-protected ring buffer for real-time audio.

    Data layout:
        buffer shape = (capacity_frames, channels)
    """

    def __init__(self, capacity_frames: int, channels: int, dtype: np.dtype):
        self.capacity = capacity_frames
        self.channels = channels
        self.dtype = dtype

        # One-time allocation, no memory is allocated during the callback
        self.buffer = np.zeros((capacity_frames, channels), dtype=dtype)
        self.write_pos = 0
        self.read_pos = 0
        self.lock = threading.Lock()

        # Statistics
        self.size = 0  # The number of valid frames in the current buffer
        self.total_written = 0   # Write frames
        self.total_read = 0      # Playback frames
        self.underrun_count = 0  # The number of playback stutters
        self.overflow_count = 0  # The number of written too fast

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
            if frames > self.capacity - self.size:
                # Insufficient buffer zone, discard excess (real-time system must make trade-offs)）
                self.overflow_count += 1
                frames = self.capacity - self.size
                data = data[:frames]
                logging.warning(f"Audio data overflow count {self.overflow_count}")

            first = min(frames, self.capacity - self.write_pos)
            second = frames - first

            self.buffer[self.write_pos:self.write_pos + first] = data[:first]
            if second > 0:
                self.buffer[0:second] = data[first:first + second]

            self.write_pos = (self.write_pos + frames) % self.capacity
            self.size += frames
            self.total_written += frames

    def read(self, frames: int) -> np.ndarray:
        """
        Read exactly `frames` frames from buffer.
        If not enough data, zeros will be returned.
        """
        out = np.zeros((frames, self.channels), dtype=self.dtype)

        with self.lock:
            if self.size < frames:
                self.underrun_count += 1

            read_frames = min(frames, self.size)
            first = min(read_frames, self.capacity - self.read_pos)
            second = read_frames - first

            out[:first] = self.buffer[self.read_pos:self.read_pos + first]
            if second > 0:
                out[first:first + second] = self.buffer[0:second]

            self.read_pos = (self.read_pos + read_frames) % self.capacity
            self.size -= read_frames
            self.total_read += read_frames

        return out

    @property
    def available_frames(self) -> int:
        """Frames currently buffered (delay)"""
        with self.lock:
            return self.size

    @property
    def written_frames(self) -> int:
        with self.lock:
            return self.total_written

    @property
    def read_frames(self) -> int:
        with self.lock:
            return self.total_read


class AudioPlayer:
    def __init__(
        self,
        samplerate: int,
        channels: int,
        capacity: int = 0,
        latency_sec: float = 0.1,
        dtype: str = 'float32',
    ):
        self.samplerate = samplerate
        self.channels = channels
        self.latency_sec = latency_sec
        self.dtype = np.dtype(dtype)
        self.blocksize = int(latency_sec * samplerate)
        self.capacity = capacity if capacity else self.blocksize * 30  # experience points
        self.buffer = AudioBuffer(self.capacity, channels, self.dtype)
        self.stream = sd.OutputStream(
            samplerate=samplerate,
            channels=channels,
            dtype=dtype,
            blocksize=self.blocksize,
            callback=self._callback,
        )

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
        if speed != 1.0:
            data_stretch = librosa.effects.time_stretch(data.T, rate=speed).T
        else:
            data_stretch = data
        self._add_data(data_stretch)

        if not self.stream.active:
            self.stream.start()

    def stop(self):
        if self.stream.active:
            self.stream.stop()

    def close(self):
        self.stop()
        self.stream.close()

    @property
    def written_seconds(self) -> float:
        """Total audio duration written to the player since start."""
        return self.buffer.written_frames / self.samplerate

    @property
    def played_second(self) -> float:
        """Total audio duration actually played."""
        return self.buffer.read_frames / self.samplerate

    @property
    def buffered_seconds(self) -> float:
        """Current buffered audio duration (latency)."""
        return self.buffer.available_frames / self.samplerate


class FilePlayer(AudioPlayer):
    def __init__(self, audio_path: str, **kwargs):
        dtype = kwargs.get('dtype', 'float32')
        self.data, sr = sf.read(audio_path, dtype=dtype)
        super().__init__(
            samplerate=sr,
            channels=self.data.shape[1] if self.data.ndim == 2 else 1,
            capacity=sr*3,
            latency_sec=0.1,
            dtype=dtype,
            **kwargs,
        )

    def play(self, *, speed: float = 1.0):
        """
        Blocking file playback with adjustable speed.

        Args:
            speed: playback speed (>1.0 faster, <1.0 slower)
        """
        chunk_size = int(self.latency_sec * self.samplerate)
        idx = 0
        total_frames = len(self.data)

        while idx < total_frames:
            chunk = self.data[idx:idx+chunk_size]

            # wait if buffer is too full
            while self.buffer.available_frames > self.buffer.capacity // 2:
                time.sleep(0.005)

            self.raw_play(chunk, speed=speed)
            idx += chunk_size

        # wait until buffer is consumed
        while self.buffer.available_frames > 0:
            time.sleep(0.01)

        self.stop()

    def get_chunks(self, *, chunk_size: int, dtype: str = 'float32') -> List[np.ndarray]:
        chunks = []
        for i in range(0, len(self.data), chunk_size):
            chunk = self.data[i:i+chunk_size]
            chunk = AudioConvert.convert(chunk, np.dtype(dtype))
            chunks.append(chunk)
        return chunks
