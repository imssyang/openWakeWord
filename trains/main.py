import os
import collections
import numpy as np
from numpy.lib.format import open_memmap
from pathlib import Path
from tqdm import tqdm
import openwakeword
import openwakeword.data
import openwakeword.utils
import openwakeword.metrics
import scipy
import datasets
import matplotlib.pyplot as plt
import torch
from torch import nn


# Download CV11 test split from HuggingFace, and convert the audio into 16 khz, 16-bit wav files
cv_11 = datasets.load_dataset("AudioLLMs/common_voice_15_en_test", "default", split="test", streaming=True)
#cv_11 = cv_11.cast_column("audio", datasets.Audio(sampling_rate=16000, num_channels=1)) # convert to 16-khz
cv_11 = iter(cv_11)

# Convert and save clips (only first 5000)
limit = 5
for i in tqdm(range(limit)):
    example = next(cv_11)
    output = os.path.join("cv11_test_clips", example["path"][0:-4] + ".wav")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    wav_data = (example["audio"]["array"]*32767).astype(np.int16) # convert to 16-bit PCM format
    scipy.io.wavfile.write(output, 16000, wav_data)
