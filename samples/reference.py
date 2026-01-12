import numpy as np
import openwakeword
import os
from typing import List, Tuple
from samples.utils import format_np_floats
from samples.utils import FilePlayer, RealPlayer


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
        print(f"Starting real-time wake word detection by {self.model_paths} ...")
        i = 0
        acc = FrameAccumulator(frame_size=1280)
        # Get audio data containing 16-bit 16khz PCM audio data from microphone.
        with RealPlayer(samplerate=16000, chunksize=512, channels=1, dtype='float32') as player:
            while True:
                data, overflowed = player.read()
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
        player = FilePlayer(audio_path, dtype="float32")
        if player.samplerate != 16000:
            raise ValueError("The model requires 16kHz, please resample first")

        acc = FrameAccumulator(frame_size=1280)
        frames = acc.add(player.data)
        last = acc.flush()
        if last is not None:
            frames.append(last)

        for i, frame in enumerate(frames):
            prediction = self.model.predict(frame)
            print(f"predict[{i}:{80*i}ms]={format_np_floats(prediction)}")

    def predict_clip(self, audio_path: str):
        # Get predictions for individual WAV files (16-bit 16khz PCM)
        predictions = self.model.predict_clip(audio_path)
        print(f"predict_clip={format_np_floats(predictions)}")

    def predict_bulk(self, audio_paths: Tuple[str]):
        # Get predictions for a large number of files using multiprocessing
        predictions = openwakeword.utils.bulk_predict(
            file_paths=audio_paths,
            wakeword_models=self.model_paths,
            ncpu=2,
        )
        print(f"predict_bulk={format_np_floats(predictions)}")


if __name__ == "__main__":
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = f"{work_dir}/openwakeword/resources/models"
    data_dir = f"{work_dir}/data"
    alexa_file = f"{data_dir}/alexa/alexa_test.wav"
    hey_jane_file = f"{data_dir}/hey_jane/hey_jane_test.wav"
    hey_mycroft_file = f"{data_dir}/hey_mycroft/hey_mycroft_test.wav"
    
    wwm = WakeWordModel([
        f"{model_dir}/alexa_v0.1.tflite",
        f"{model_dir}/hey_jarvis_v0.1.tflite",
        f"{model_dir}/hey_mycroft_v0.1.tflite",
    ])
    wwm.predict_microphone()
    #wwm.predict_file(alexa_file)
    #wwm.predict_file(hey_jane_file)
    #wwm.predict_file(hey_mycroft_file)
    #wwm.predict_clip(alexa_file)
    #wwm.predict_clip(hey_jane_file)
    #wwm.predict_clip(hey_mycroft_file)
    #wwm.predict_bulk([alexa_file, hey_jane_file, hey_mycroft_file])
