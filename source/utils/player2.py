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


class FilePlayer(AudioSound):
    def __init__(self, path: str, dtype: str = 'float32'):
        self.data, self.samplerate = sf.read(path, dtype=dtype)
        print(f"File {path}: {self.samplerate}Hz with {type(self.data)}/{self.data.shape}")




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
