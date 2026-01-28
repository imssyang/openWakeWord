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
    pass


def test_mixing():
    primary_path = f"{work_dir}/data/wakeword/turn_on_the_office_lights/verifier/turn_on_the_office_lights_test.wav"
    secondary_path = f"{work_dir}/data/wakeword/turn_on_the_office_lights/verifier/santa_barbara_corpus_test.wav"
    output_path = f"{work_dir}/data/wakeword/turn_on_the_office_lights/verifier/maxing.wav"
    primary, primary_sr = sf.read(primary_path)
    secondary, secondary_sr = sf.read(secondary_path)
    output_sr = 16000
    mixing = AudioConvert.mixing(
        primary,
        primary_sr,
        secondary=secondary,
        secondary_sr=secondary_sr,
        output_sr=output_sr,
        start_sec=0.0,
        duration_sec=20,
    )
    sf.write(output_path, mixing, output_sr)
