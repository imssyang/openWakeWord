import numpy as np
import sounddevice as sd
import soundfile as sf
import threading


class AudioPlayer:
    @classmethod
    def list_devices(cls):
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            print(
                f"{idx}: {device['name']} - PulseAudio "
                f"channels ({device['max_input_channels']} in, {device['max_output_channels']} out)"
            )

    @classmethod
    def play_data(cls, data, samplerate, dtype='float32'):
        data = data.astype(np.float32)
        channels = data.shape[1] if data.ndim > 1 else 1
        with sd.OutputStream(samplerate=samplerate, channels=channels, dtype=dtype) as stream:
            stream.write(data)

    @classmethod
    def play_data2(cls, data, samplerate, dtype='float32'):
        done = threading.Event()
        data = data.astype(np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        channels = data.shape[1]
        cb_pos = 0

        def callback(outdata, frames, time, status):
            if status:
                print(f"Err with status: {status}", file=sys.stderr)
            
            nonlocal cb_pos
            start = cb_pos
            end = start + frames
            chunk = data[start:end]
            if len(chunk) < frames:
                outdata[:len(chunk)] = chunk
                outdata[len(chunk):] = 0
                done.set()
            else:
                outdata[:] = chunk
            cb_pos = end

        with sd.OutputStream(samplerate=samplerate, channels=channels, dtype=dtype, callback=callback):
            done.wait()

    @classmethod
    def gen_sine_wave(cls, sample_rate: int = 44100, duration_sec: float = 1.0, frequency: float = 440.0):
        # Generate array with duration*sample_rate steps, ranging between 0 and duration
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
        x = 0.5 * np.sin(2 * np.pi * frequency * t) # Generate a 440 Hz sine wave
        print(f"Playing sine wave at {sample_rate} Hz for {duration_sec} seconds")
        return x, sample_rate

    @classmethod
    def play_sine(cls):
        data, samplerate = cls.gen_sine_wave()
        cls.play_data(data, samplerate)
        print("Sine Over")


class FilePlayer(AudioPlayer):
    def __init__(self, path: str, dtype: str = 'float32'):
        self.data, self.samplerate = sf.read(path, dtype=dtype)
        print(f"File {path}: {self.samplerate}Hz with {type(self.data)}/{self.data.shape}")

    def play(self):
        self.play_data2(self.data, self.samplerate)


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
