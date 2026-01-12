import numpy as np
import scipy.io.wavfile
import openwakeword
import os


class TrainVerifierModel:
    def __init__(self, workdir: str = "examples/verifier_data"):
        self.workdir = workdir
        self.positive_dir = os.path.join(self.workdir, "positive")
        self.negative_dir = os.path.join(self.workdir, "negative")
        os.makedirs(self.positive_dir, exist_ok=True)
        os.makedirs(self.negative_dir, exist_ok=True)
        print(f"Training custom verifier model in {self.workdir}")

    def random_negative_file(self):
        file_path = os.path.join(self.negative_dir, "reference.wav")
        scipy.io.wavfile.write(
            file_path,
            16000,
            np.random.randint(-1000, 1000, 16000*4).astype(np.int16),
        )
        return file_path

    def train(self):
        # Load random clips
        positive_clips = [os.path.join(self.positive_dir, "hey_mycroft_test.wav")]
        negative_clips = [self.random_negative_file()]

        # Train verifier model on the reference clips, using full path of model file
        openwakeword.train_custom_verifier(
            positive_reference_clips=positive_clips,
            negative_reference_clips=negative_clips,
            output_path=os.path.join(self.workdir, 'verifier_model.pkl'),
            model_name=os.path.join("openwakeword", "resources", "models", "hey_mycroft_v0.1.tflite")
        )

        # Load model with verifier model incorrectly to catch ValueError
        owwModel = openwakeword.Model(
            wakeword_models=[os.path.join("openwakeword", "resources", "models", "hey_mycroft_v0.1.tflite")],
            custom_verifier_models={"hey_mycroft_v0.1": os.path.join(self.workdir, "verifier_model.pkl")},
            custom_verifier_threshold=0.3,
        )

        # Prediction on random data
        predictions = owwModel.predict_clip(positive_clips[0])
        print(f"Predictions with custom verifier model: {predictions}")


if __name__ == "__main__":
    TrainVerifierModel().train()
