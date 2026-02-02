import numpy as np
import pytest
import sounddevice as sd
import soundfile as sf
import tempfile
import time
from source.utils import AudioPlayer, FilePlayer, MicPlayer


def sine_wave(sample_rate: int, channels: int, duration_sec: float, frequency: float):
    # Generate array with duration*sample_rate steps, ranging between 0 and duration
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * frequency * t) # Generate sine wave
    x = np.repeat(x[:, None], channels, axis=1)
    print(f"Playing sine wave at {sample_rate} Hz channels {channels} for {duration_sec} seconds")
    return x


def test_devices():
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        print(
            f"{idx}: {device['name']} - PulseAudio "
            f"channels ({device['max_input_channels']} in, {device['max_output_channels']} out)"
        )


def test_audio_basic():
    samplerate = 16000
    channels = 2
    with AudioPlayer(samplerate=samplerate, channels=channels) as player:
        # Generate 0.5 second test audio (sine wave)
        test_data = sine_wave(samplerate, channels, 0.5, 440)
        player.raw_play(test_data)
        time.sleep(0.2)  # waiting for the callback will read the ring buffer

    assert player.buffer.written_second >= 0
    assert player.buffer.played_second >= 0
    assert player.buffer.available_second >= 0
    assert player.buffer.underrun_count >= 0
    assert player.buffer.overflow_count >= 0
    player.close()


def test_audio_mono():
    samplerate = 8000
    channels = 1
    with AudioPlayer(samplerate=samplerate, channels=channels) as player:
        # Mono audio
        test_data = sine_wave(samplerate, channels, 0.2, 300)
        player.raw_play(test_data)
        time.sleep(0.1)

    assert player.buffer.played_second > 0
    player.close()


def test_audio_file():
    samplerate = 16000
    channels = 2
    duration_sec = 0.6
    test_data = sine_wave(samplerate, channels, duration_sec, 440)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, test_data, samplerate)
        path = f.name
        player = FilePlayer(path)

        assert player.samplerate == samplerate
        assert player.channels == channels
        assert player.dtype == "float32"

        speed = 1.0
        player.play(speed=speed)
        assert player.buffer.played_second == pytest.approx(duration_sec/speed, rel=0.01)
        assert player.buffer.written_second == pytest.approx(duration_sec/speed, rel=0.01)
        assert player.buffer.available_second == pytest.approx(0, rel=0.00001)

        speed = 3.0
        player.play(speed=speed)
        assert player.buffer.played_second == pytest.approx(duration_sec/speed, rel=0.01)
        assert player.buffer.written_second == pytest.approx(duration_sec/speed, rel=0.01)
        assert player.buffer.available_second == pytest.approx(0, rel=0.00001)

        player.close()


@pytest.mark.integration
def test_audio_mic():
    with MicPlayer(
        samplerate=16000,
        channels=1,
        latency_sec=0.05,
    ) as player:
        time.sleep(3.0)

    # At least some data should go into the buffer
    assert player.buffer.available_size > 0
    player.close()
