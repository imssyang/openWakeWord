import os
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import torch
from source.utils import AudioPlayer, AudioFile, FilePlayer, MicPlayer


work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_vad_torch():
    os.environ["TORCH_HOME"] = f"{work_dir}/models/torch"
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True,
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    print(f"Loaded model: {model}", utils)
    
    confidences = []
    samplerate = 16000
    chunks = AudioFile(f"{work_dir}/data/vad/test.wav").get_chunks(
        chunksize=512, samplerate=samplerate, enable_mono=True, dtype='float32')
    for i, chunk in enumerate(chunks):
        chunk = chunk.flatten()  # shape: (512,)

        with torch.no_grad():
            confidence = model(
                torch.from_numpy(chunk),
                samplerate,
            ).item()

        confidences.append(confidence)

    print("Stopped Recording")
    plt.figure(figsize=(20, 6))
    plt.plot(confidences)
    plt.title("Voiced Confidence")
    plt.xlabel("Frame")
    plt.ylabel("Confidence")
    plt.savefig(f"{work_dir}/tests/test_vad_output.png")
    plt.close()

