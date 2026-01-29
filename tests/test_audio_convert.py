import numpy as np
import os
import soundfile as sf
from source.utils.convert import AudioConvert


work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_uint8():
    u0 = np.array([0, 128, 255], dtype=np.uint8)
    assert np.allclose(
        AudioConvert.to_float32(u0),
        np.array([-1.0, 0.0, 127/128], dtype=np.float32),
        atol=1e-6,
    )
    assert np.array_equal(
        AudioConvert.to_int16(u0),
        np.array([-32767, 0, int(127/128*32767)], dtype=np.int16),
    )


def test_float32():
    f1 = np.array([-1, 1], dtype=np.float32)
    assert np.allclose(
        AudioConvert.to_int16(f1),
        np.array([-32767, 32767], dtype=np.int16),
    )


def test_convert():
    f0 = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32)
    assert np.allclose(
        AudioConvert.convert(f0, np.uint8),
        np.array([0, 128, 192, 255], dtype=np.uint8),
    )


def test_transform():
    sr_in = 44100
    sr_out = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr_in * duration), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    stereo = np.stack([left, right], axis=0)  # (2, N)
    mono = AudioConvert.transform(
        stereo,
        orig_sr=sr_in,
        target_sr=sr_out,
        enable_mono=True,
    )
    assert mono.ndim == 1
    assert abs(len(mono) - int(sr_out * duration)) <= 2
    assert np.all(np.isfinite(mono))
    assert np.max(np.abs(mono)) <= 1.0
    assert np.mean(mono**2) > 1e-6


def test_mixing():
    sr_primary = 16000
    sr_secondary = 44100
    output_sr = 16000
    duration_primary = 10.0  # seconds
    duration_mix = 3.0
    start_sec = 2.0

    # Construct primary audio
    t_p = np.linspace(0, duration_primary, int(sr_primary * duration_primary), endpoint=False)
    primary = 0.3 * np.sin(2 * np.pi * 440 * t_p)

    # Construct secondary audio (shorter, different sr)
    t_s = np.linspace(0, duration_mix, int(sr_secondary * duration_mix), endpoint=False)
    secondary = 0.3 * np.sin(2 * np.pi * 880 * t_s)

    # Run mixing
    mixed = AudioConvert.mixing(
        primary=primary,
        primary_sr=sr_primary,
        secondary=secondary,
        secondary_sr=sr_secondary,
        secondary_gain=0.5,
        output_sr=output_sr,
        start_sec=start_sec,
        duration_sec=duration_mix,
        loop_secondary=False,
        enable_mono=True,
    )

    # shape / sr invariants
    assert mixed.ndim == 1
    assert abs(len(mixed) - len(primary)) <= 2

    # numeric safety
    assert np.all(np.isfinite(mixed))
    assert np.max(np.abs(mixed)) <= 1.0

    # energy checks (mixing area should be significantly more energy)
    start = int(start_sec * output_sr)
    end = start + int(duration_mix * output_sr)
    energy_before = np.mean(primary[start:end] ** 2)
    energy_after = np.mean(mixed[start:end] ** 2)
    assert energy_after > energy_before * 1.2

    # non-mixing region unchanged
    assert np.allclose(
        mixed[:start],
        primary[:start],
        atol=1e-4,
    )
