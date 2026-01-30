import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
from .convert import AudioConvert


class AudioSound(AudioConvert):
    @classmethod
    def play_data(cls, data, samplerate, dtype='float32'):
        data = data.astype(np.float32)
        channels = data.shape[1] if data.ndim > 1 else 1
        with sd.OutputStream(samplerate=samplerate, channels=channels, dtype=dtype) as stream:
            stream.write(data)

    @classmethod
    def load_file(cls, audio_path: str, sample_rate: int, enable_mono: bool, dtype: str = 'float32'):
        audio_data, sr = sf.read(audio_path)
        audio_target = cls.transform(audio_data, orig_sr=sr, target_sr=sample_rate, enable_mono=enable_mono)
        if dtype in ["int16"]:
            audio_target = np.clip(audio_target, -1.0, 1.0)
            return (audio_target * 32767.0).astype(np.int16)
        else:
            return audio_target.astype(dtype)


class FilePlayer(AudioSound):
    def __init__(self, path: str, dtype: str = 'float32'):
        self.data, self.samplerate = sf.read(path, dtype=dtype)
        print(f"File {path}: {self.samplerate}Hz with {type(self.data)}/{self.data.shape}")

    def play(self):
        self.play_data2(self.data, self.samplerate)

    def get_chunks(self, chunk_size: int, dtype: np.dtype):
        chunks = []
        num_frames = len(self.data) // chunk_size
        for i in range(num_frames):
            chunk = audio[i*chunk_size : (i+1)*chunk_size]
            chunks.append(AudioConvert.convert(chunk, dtype))
        return chunks
    

class RealPlayer:
    def __init__(self, samplerate=16000, chunksize=512, channels=1, dtype='float32'):
        self.chunksize = chunksize
        self.stream = sd.RawInputStream(
            samplerate=samplerate,
            blocksize=chunksize,
            channels=channels,
            dtype=dtype,
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self.stream.start()

    def read(self):
        if self.stream is None or not self.stream.active:
            raise RuntimeError("Audio stream not started")
        return self.stream.read(self.chunksize)

    def stop(self):
        self.stream.stop()
        self.stream.close()
