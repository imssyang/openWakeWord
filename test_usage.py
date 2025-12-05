import openwakeword
from openwakeword.model import Model
from openwakeword.utils import bulk_predict
import soundfile as sf


model = Model(
    wakeword_models=[
        "openwakeword/resources/models/alexa_v0.1.tflite"
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
print(f"{prediction=}")

# Get predictions for individual WAV files (16-bit 16khz PCM)
predictions = model.predict_clip("tests/data/alexa_test.wav")
print(f"{predictions=}")

# Get predictions for a large number of files using multiprocessing
predictions2 = bulk_predict(
    file_paths = ["tests/data/alexa_test.wav", "tests/data/hey_jane.wav"],
    wakeword_models = ["hey jarvis"],
    ncpu=2
)
print(f"{predictions2=}")
