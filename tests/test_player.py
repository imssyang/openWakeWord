from samples.utils import (
    AudioPlayer,
    FilePlayer,
    RealPlayer,
)


def test_sine():
    AudioPlayer.list_devices()
    AudioPlayer.play_sine()


def test_file():
    FilePlayer("data/alexa/alexa_test.wav").play()

