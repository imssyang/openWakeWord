import sounddevice as sd
import numpy as np
import torch
import matplotlib.pyplot as plt
from source.utils import (
    AudioSound,
    FilePlayer,
    RealPlayer,
)


def test_audio():
    fp = FilePlayer("data/vad/test.wav", dtype="int16")
    chunks = fp.get_chunks(chunk_size=512, dtype=np.int16)
    for i, chunk in enumerate(chunks):
        pass

SAMPLE_RATE = 16000
NUM_SAMPLES = 512
FRAMES_TO_RECORD = 50

data = []
voiced_confidences = []

print("Started Recording")

for _ in range(FRAMES_TO_RECORD):
    # 读取一段音频（阻塞）
    audio_chunk, _ = sd.rec(
        frames=NUM_SAMPLES,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocking=True,
    )

    audio_chunk = audio_chunk.flatten()  # shape: (512,)

    # 保存原始音频
    data.append(audio_chunk.copy())

    # int16 -> float32
    audio_float32 = audio_chunk.astype(np.float32) / 32768.0

    # 推理
    with torch.no_grad():
        confidence = model(
            torch.from_numpy(audio_float32),
            SAMPLE_RATE
        ).item()

    voiced_confidences.append(confidence)

print("Stopped Recording")

# plot
plt.figure(figsize=(20, 6))
plt.plot(voiced_confidences)
plt.title("Voiced Confidence")
plt.xlabel("Frame")
plt.ylabel("Confidence")
plt.show()
