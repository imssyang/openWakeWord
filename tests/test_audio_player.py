import numpy as np
import pytest
from time import sleep
from source.utils import AudioPlayer, FilePlayer


def test_audio_basic():
    samplerate = 16000
    channels = 2
    player = AudioPlayer(samplerate=samplerate, channels=channels)

    # Generate 0.5 second test audio (sine wave)
    t = np.linspace(0, 0.5, int(0.5 * samplerate), endpoint=False)
    freq = 440
    test_data = 0.1 * np.sin(2 * np.pi * freq * t)
    test_data = np.repeat(test_data[:, None], channels, axis=1)  # stereo
    player.raw_play(test_data)
    sleep(0.2)  # waiting for the callback will read the ring buffer

    assert player.written_seconds >= 0
    assert player.played_second >= 0
    assert player.buffered_seconds >= 0
    assert player.buffer.underrun_count >= 0
    assert player.buffer.overflow_count >= 0

    player.close()


def test_audio_mono():
    samplerate = 8000
    channels = 1
    player = AudioPlayer(samplerate=samplerate, channels=channels)

    # Mono audio
    t = np.linspace(0, 0.2, int(0.2 * samplerate), endpoint=False)
    test_data = 0.2 * np.sin(2 * np.pi * 300 * t)
    test_data = test_data[:, None]
    player.raw_play(test_data)
    sleep(0.1)

    assert player.played_second > 0
    player.stop()
    player.close()


def test_file_player():
    FilePlayer("/opt/ai/openWakeWord/data/music/fma_sample/000002.wav").play(speed=1.0)


