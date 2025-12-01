import openwakeword
from openwakeword.model import Model
import soundfile as sf

model = Model(
    wakeword_models=[
        "/usr/local/lib/python3.10/dist-packages/openwakeword/resources/models/alexa_v0.1.onnx"
    ],  # can also leave this argument empty to load all of the included pre-trained models
)


def get_audio_frame():
    audio, sr = sf.read("tests/data/alexa_test.wav")    # audio → np.ndarray, sr → 采样率
    print(audio.shape)
    print(type(audio))
    return audio


# Get audio data containing 16-bit 16khz PCM audio data from a file, microphone, network stream, etc.
# For the best efficiency and latency, audio frames should be multiples of 80 ms, with longer frames
# increasing overall efficiency at the cost of detection latency
frame = get_audio_frame()


# Get predictions for the frame
prediction = model.predict(frame)
print(prediction)
