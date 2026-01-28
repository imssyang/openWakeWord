import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
from .convert import AudioConvert


class AudioSound(AudioConvert):
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

    @classmethod
    def mix_music(
        cls,
        input_path: str,
        music_path: str,
        output_path: str,
        sample_rate: int,
        enable_mono: bool,
        mix_sec: int,
        music_gain: float = 0.7,
        dtype: str = 'float32',
    ):
        dat, sr = sf.read(input_path, dtype=dtype)
        dat_music, sr_music = sf.read(music_path, dtype=dtype)
        tdat = cls.transform(dat, orig_sr=sr, target_sr=sample_rate, enable_mono=enable_mono)
        tdat_music = cls.transform(dat_music, orig_sr=sr_music, target_sr=sample_rate, enable_mono=enable_mono)
        mix_len = mix_sec * sample_rate
        speech_tail = dat[-mix_len:]     # Take tail voice
        music_head = dat_music[:mix_len] # Take head music
        mixed_tail = (speech_tail + music_gain * music_head) * 0.5  # Mix (music attenuation 0.7, then averaged)
        tdat[-mix_len:] = mixed_tail     # Write back to the original audio
        sf.write(output_path, tdat, sample_rate)

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
