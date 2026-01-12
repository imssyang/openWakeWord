import collections
import datasets
import io
import os
import kagglehub
import librosa
import matplotlib.pyplot as plt
import numpy as np
import openwakeword
import pandas as pd
import scipy
import soundfile as sf
import torch
from tqdm import tqdm


class CV17Dataset:
    def __init__(
        self,
        hf_path: str,
        n_elements: int,
    ):
        self.output_dir = f"{hf_path}/common_voice_17"
        self.cache_dir = f"{hf_path}/_cache"
        self.dataset = datasets.load_dataset(
            "fixie-ai/common_voice_17_0",
            "en",
            split="train",
            streaming=True,
            cache_dir=self.cache_dir,
        ).take(n_elements)

    def download(self, target_sr: int, enable_mono: bool):
        os.makedirs(self.output_dir, exist_ok=True)
        for i, element in enumerate(self.dataset):
            sentence = element['sentence']
            audio_path = element['audio']['path']
            audio_data = element['audio']['array']
            audio_sr = element['audio']['sampling_rate']
            print(f"Index[{i}] {audio_path=} {audio_data.shape=} {audio_sr=} {sentence=}")

            audio_target = librosa.resample(
                librosa.to_mono(audio_data) if enable_mono else audio_data,
                orig_sr=audio_sr,
                target_sr=target_sr,
            )
            
            audio_base, _ = os.path.splitext(audio_path)
            save_path = os.path.join(self.output_dir, f"cv17_{i}_{audio_base}.wav")
            sf.write(save_path, audio_target, target_sr)


if __name__ == "__main__":
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hf_dir = f"{work_dir}/data/huggingface"
    cv17 = CV17Dataset(hf_path=hf_dir, n_elements=10)
    cv17.download(target_sr=16000, enable_mono=True)


#final_ds = Dataset.from_list(list(small_ds))
#final_ds.save_to_disk("./cv_100_samples")
