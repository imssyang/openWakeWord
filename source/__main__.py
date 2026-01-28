import os
import torch
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API.*"
)

from .dataset import CV17Dataset
from .training import OWWDataset, OWWModel
from .reference import WakeWordModel


class OWWMain:
    def __init__(self):
        self.wake_word = "turn_on_the_office_lights"
        self.work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = f"{self.work_dir}/data"
        self.cv17_dir = f"{self.data_dir}/huggingface/common_voice_17"
        self.music_dir = f"{self.data_dir}/music/fma_sample"
        self.wakeword_dir = f"{self.data_dir}/wakeword/{self.wake_word}"
        self.feature_dir = f"{self.wakeword_dir}/feature"
        self.positive_dirs = [
            f"{self.wakeword_dir}/positive"
        ]
        self.negative_dirs = [
            f"{self.cv17_dir}/en/train",
            f"{self.cv17_dir}/en/test",
        ]
        self.positive_feature_path = f"{self.feature_dir}/positive.npy"
        self.negative_feature_path = f"{self.feature_dir}/negative.npy"
        self.sample_rate = 16000
        self.enable_mono = True
        self.window_sec = 3
        self.num_epochs = 10
        self.batch_size = 512

    def download_cv17(self):
        CV17Dataset(
            root_dir=self.cv17_dir,
            language="en",
            n_train=5000,
            n_test=50,
            sample_rate=self.sample_rate,
            enable_mono=self.enable_mono,
        )

    def save_features(self):
        oww_dataset = OWWDataset(
            window_sec=self.window_sec,
            sample_rate=self.sample_rate,
            enable_mono=self.enable_mono,
        )
        oww_dataset.save_negative_features(
            self.negative_feature_path,
            negative_dirs=self.negative_dirs,
        )
        oww_dataset.save_positive_features(
            self.positive_feature_path,
            positive_dirs=self.positive_dirs,
            negative_dirs=self.negative_dirs,
        )

    def train_model(self):
        oww_dataset = OWWDataset(
            window_sec=self.window_sec,
            sample_rate=self.sample_rate,
            enable_mono=self.enable_mono,
        )
        oww_dataset.load_features(
            positive_feature_path=self.positive_feature_path,
            negative_feature_path=self.negative_feature_path,
            batch_size = self.batch_size,
        )
        oww_dataset.show()

        oww_model = OWWModel(
            oww_dataset,
            layer_dim=32,
            learn_rate=0.001,
        )
        oww_model.train(num_epochs=self.num_epochs)
        oww_model.save_onnx(f"{self.wakeword_dir}/{self.wake_word}.onnx")
        oww_model.plot_metrics(f"{self.wakeword_dir}/train_epoch{self.num_epochs}.png")

        scores = oww_model.predict(f"{self.wakeword_dir}/verifier/{self.wake_word}_test.wav")
        oww_model.plot_scores(f"{self.wakeword_dir}/verifier/{self.wake_word}_test.png", scores)

        scores_onnx = oww_model.onnx_predict(
            f"{self.wakeword_dir}/{self.wake_word}.onnx",
            f"{self.wakeword_dir}/verifier/{self.wake_word}_test.wav",
        )
        oww_model.plot_scores(f"{self.wakeword_dir}/verifier/{self.wake_word}_test_onnx.png", scores_onnx)

        #scores_music = oww_model.predict_with_mixmusic(
        #    f"{self.wakeword_dir}/verifier/{self.wake_word}_test.wav",
        #    f"{self.music_dir}/000182.wav",
        #)
        #oww_model.plot_scores(f"{self.wakeword_dir}/verifier/{self.wake_word}_test_music.png", scores_music)

    def predict_clip(self):
        wwm = WakeWordModel(
            [f"{self.wakeword_dir}/{self.wake_word}.onnx"],
            inference_framework="onnx",
            vad_threshold=0.5,
            enable_noise_suppression=True,
        )
        clips = wwm.predict_file(f"{self.wakeword_dir}/verifier/{self.wake_word}_test.wav")
        predicts = [p["turn_on_the_office_lights"] for p in clips]
        scores = []
        for p in predicts:
            logits = torch.Tensor([[p]])
            prob = torch.sigmoid(logits)
            scores.append(float(prob.item()))
        wwm.plot_scores(f"{self.wakeword_dir}/verifier/{self.wake_word}_test2.png", scores)

    def predict_clip2(self):
        audio_file = "santa_barbara_corpus_test.wav"
        wwm = WakeWordModel(
            [f"{self.wakeword_dir}/{self.wake_word}.onnx"],
            inference_framework="onnx",
            vad_threshold=0.5,
            enable_noise_suppression=True,
        )
        clips = wwm.predict_file(f"{self.wakeword_dir}/verifier/{audio_file}")
        predicts = [p["turn_on_the_office_lights"] for p in clips]
        scores = []
        for p in predicts:
            logits = torch.Tensor([[p]])
            prob = torch.sigmoid(logits)
            scores.append(float(prob.item()))
        wwm.plot_scores(f"{self.wakeword_dir}/verifier/{audio_file}.png", scores)


if __name__ == "__main__":
    oww = OWWMain()
    #oww.download_cv17()
    #oww.save_features()
    #oww.train_model()
    #oww.predict_clip()
    oww.predict_clip2()
