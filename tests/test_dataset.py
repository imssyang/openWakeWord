from samples.training import (
    AudioDataset,
    CV17Dataset,
)


def test_audio_dataset():
    for i, item in enumerate(AudioDataset(
        "data/alexa", 16000, True,
    )):
        if i >= 2:
            break
        audio = item["audio"]
        print(f"Item {i}: {audio}")
        assert "path" in audio
        assert "sampling_rate" in audio
        assert "array" in audio


def test_cv17_dataset():
    dataset = CV17Dataset(
        hf_path="data/huggingface",
        n_train=6,
        n_test=2,
        batch_size=4,
        num_workers=0,
    )
    for batch in dataset.trainloader:
        print(batch)
        print(batch["waveforms"].shape)
    for batch in dataset.testloader:
        print(batch)
