from source.utils import (
    AudioSound,
    FilePlayer,
    RealPlayer,
)


def test_sine():
    AudioSound.list_devices()
    AudioSound.play_sine()


def test_file():
    FilePlayer("data/alexa/alexa_test.wav").play()

