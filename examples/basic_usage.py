import numpy as np
import openwakeword
import sounddevice as sd
import soundfile as sf
from typing import List, Tuple


def format_np_floats(obj, decimals=6):
    if isinstance(obj, dict):
        return {k: format_np_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [format_np_floats(v, decimals) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(format_np_floats(v, decimals) for v in obj)
    elif isinstance(obj, (np.float32, np.float64, float)):
        return f"{float(obj):.{decimals}f}"
    else:
        return obj


class AudioPlayer:
    @classmethod
    def list_devices(cls):
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            print(f"{idx}: {device['name']} - PulseAudio channels ({device['max_input_channels']} in, {device['max_output_channels']} out)")

    @classmethod
    def sine_wave(cls):
        fs = 44100  # Sample rate
        duration = 1.0  # seconds
        f = 440.0  # Sound frequency (Hz)
        # Generate array with duration*fs steps, ranging between 0 and duration
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        # Generate a 440 Hz sine wave
        x = 0.5 * np.sin(2 * np.pi * f * t)
        print(f"Playing sine wave at {fs} Hz for {duration} seconds")
        return x, fs

    @classmethod
    def real_stream(self, samplerate=16000, chunksize=512, channels=1, dtype='int16'):
        stream = sd.RawInputStream(
            samplerate=samplerate,
            blocksize=chunksize,
            channels=channels,
            dtype=dtype,
        )
        stream.start()
        return stream

    @classmethod
    def load_file(cls, filename: str):
        data, samplerate = sf.read(filename, dtype='int16')
        print(f"Loaded {filename} at {samplerate} Hz data:{type(data)}/{data.shape}")
        return data, samplerate

    @classmethod
    def play(cls, data: np.ndarray, samplerate: int):
        sd.play(data, samplerate)
        sd.wait()

    @classmethod
    def play_sine(cls):
        data, samplerate = cls.sine_wave()
        cls.play(data, samplerate)

    @classmethod
    def play_file(cls, filename: str = "tests/data/alexa_test.wav"):
        data, samplerate = cls.load_file(filename)
        cls.play(data, samplerate)


class FrameAccumulator:
    def __init__(self, frame_size, dtype=np.int16):
        self.frame_size = frame_size
        self.buffer = np.zeros(0, dtype=dtype)
        self.dtype = dtype

    def add(self, data):
        """
        Add audio data of any length (np.ndarray)
        Returns 0 or more frames, each length of frame_size
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("data must be np.ndarray")

        self.buffer = np.concatenate((self.buffer, data))

        frames = []
        while len(self.buffer) >= self.frame_size:
            frame = self.buffer[:self.frame_size]
            frames.append(frame.copy())
            self.buffer = self.buffer[self.frame_size:]
        return frames

    def flush(self):
        """
        Return to the last frame (if it is insufficient, make up to frame_size)
        """
        if len(self.buffer) == 0:
            return None

        pad_len = self.frame_size - len(self.buffer)
        frame = np.pad(self.buffer, (0, pad_len), mode='constant')
        self.buffer = np.zeros(0, dtype=self.dtype)
        return frame


class WakeWordModel:
    def __init__(self, model_paths: List[str]):
        #np.set_printoptions(precision=6, suppress=True)
        self.model_paths = model_paths
        self.model = openwakeword.model.Model(wakeword_models=model_paths)

    def predict_microphone(self):
        # Get audio data containing 16-bit 16khz PCM audio data from microphone.
        chunksize = 512  # The microphone may return any length
        stream = AudioPlayer.real_stream(chunksize=chunksize)
        acc = FrameAccumulator(frame_size=1280)
        print(f"Starting real-time wake word detection by {self.model_paths} ...")
        i = 0
        while True:
            data, overflowed = stream.read(chunksize)
            audio = np.frombuffer(data, dtype=np.int16)
            frames = acc.add(audio)
            for frame in frames:
                i += 1
                prediction = self.model.predict(frame)
                print(f"predict_real[{i}:{80*i}ms]={format_np_floats(prediction)}")

    def predict_file(self, audio_path: str):
        # Get audio data containing 16-bit 16khz PCM audio data from a file, microphone, network stream, etc.
        # For the best efficiency and latency, audio frames should be multiples of 80 ms, with longer frames
        # increasing overall efficiency at the cost of detection latency
        data, samplerate = AudioPlayer.load_file(audio_path)
        if samplerate != 16000:
            raise ValueError("The model requires 16kHz, please resample first")

        acc = FrameAccumulator(frame_size=1280)
        frames = acc.add(data)
        last = acc.flush()
        if last is not None:
            frames.append(last)

        for i, frame in enumerate(frames):
            prediction = self.model.predict(frame)
            print(f"predict[{i}:{80*i}ms]={format_np_floats(prediction)}")

    def predict_clip(self, audio_path: str):
        # Get predictions for individual WAV files (16-bit 16khz PCM)
        predictions = self.model.predict_clip(audio_path)
        print(f"predict_clip={predictions}")

    def bulk_predict(self, audio_paths: Tuple[str]):
        # Get predictions for a large number of files using multiprocessing
        predictions = openwakeword.utils.bulk_predict(
            file_paths=audio_paths,
            wakeword_models=["hey jarvis"],
            ncpu=2,
        )
        print(f"bulk_predict={predictions}")


if __name__ == "__main__":
    AudioPlayer.list_devices()
    #AudioPlayer.play_sine()
    #AudioPlayer.play_file()
    wwm = WakeWordModel([
        "openwakeword/resources/models/alexa_v0.1.tflite",
    ])
    #wwm.predict_microphone()
    wwm.predict_file("tests/data/alexa_test.wav")
    wwm.predict_file("tests/data/hey_jane.wav")
    wwm.predict_file("tests/data/hey_mycroft_test.wav")
    #wwm.predict_clip("tests/data/alexa_test.wav")
    #wwm.predict_clip("tests/data/hey_jane.wav")
    #wwm.predict_clip("tests/data/hey_mycroft_test.wav")
    #wwm.bulk_predict(["tests/data/alexa_test.wav", "tests/data/hey_jane.wav"])
