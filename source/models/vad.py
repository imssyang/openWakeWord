# This file contains the implementation of a class for voice activity detection (VAD),
# based on the pre-trained model from Silero (https://github.com/snakers4/silero-vad).
# It can be used as with the openWakeWord library, or independently.
import logging
import numpy as np
import onnxruntime as ort
import os
from collections import deque


class SileroVAD:
    """
    A model class for a voice activity detection (VAD) based on Silero's model:

    https://github.com/snakers4/silero-vad
    """
    def __init__(
        self,
        model_path: str = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "resources",
            "models",
            "silero_vad.onnx"
        ),
        n_threads: int = 1,
    ):
        # Initialize the ONNX model
        sessionOptions = ort.SessionOptions()
        sessionOptions.inter_op_num_threads = n_threads
        sessionOptions.intra_op_num_threads = n_threads
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sessionOptions,
            providers=["CPUExecutionProvider"],
        )

        # Reset model to start
        self.reset_states()

        if '16k' in model_path:
            logging.warning('This model support only 16000 sampling rate!')
            self.sample_rates = [16000]
        else:
            self.sample_rates = [8000, 16000]

    def _validate_input(self, x: np.ndarray, sr: int):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        if x.ndim > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.ndim}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:,::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)")
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")
        return x, sr

    def reset_states(self, batch_size=1):
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros(0, dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    def process_chunk(self, x: np.ndarray, sr: int) -> np.ndarray:
        x, sr = self._validate_input(x, sr)

        num_samples = 512 if sr == 16000 else 256
        if x.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {x.shape[-1]} "
                f"(Supported values: 256 for 8000 sample rate, 512 for 16000)"
            )

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if self._last_sr and self._last_sr != sr:
            self.reset_states(batch_size)
        if self._last_batch_size and self._last_batch_size != batch_size:
            self.reset_states(batch_size)

        if self._context.size == 0:
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        x = np.concatenate([self._context, x], axis=1)
        ort_inputs = {'input': x, 'state': self._state, 'sr': np.array(sr, dtype='int64')}
        ort_outs = self.session.run(None, ort_inputs)
        out, self._state = ort_outs
        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size
        return out

    def process_audio(self, x: np.ndarray, sr: int) -> np.ndarray:
        x, sr = self._validate_input(x, sr)
        self.reset_states()

        num_samples = 512 if sr == 16000 else 256
        if x.shape[1] % num_samples != 0:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = np.pad(
                x, 
                pad_width=((0, 0), (0, pad_num)),
                mode='constant', 
                constant_values=0.0,
            )

        outs = []
        for i in range(0, x.shape[1], num_samples):
            wavs_batch = x[:, i:i+num_samples]
            out_chunk = self.process_chunk(wavs_batch, sr)
            outs.append(out_chunk)

        stacked = np.concatenate(outs, axis=1)
        return stacked
