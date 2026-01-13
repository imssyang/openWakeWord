import datasets
import numpy as np
import openwakeword.data
from samples.utils import AudioPlayer
from samples.training import (
    AudioDataset,
    CV17Dataset,
)
from tqdm import tqdm


def test_audio_dataset():
    for i, item in enumerate(AudioDataset(
        ["data/alexa"], 16000, True,
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
        batch_size=4,
        n_train=6,
        n_test=2,
    )
    for batch in dataset.trainloader:
        print(batch)
        print(batch["waveforms"].shape)
    for batch in dataset.testloader:
        print(batch)


def test_filter_paths():
    # Get example paths, filtering out clips that are too long or too short
    paths, durations = openwakeword.data.filter_audio_paths(
        [
            #"fma_sample",
            #"fsd50k_sample",
            "data/huggingface/common_voice_17/en/train",
            "data/huggingface/common_voice_17/en/test",
        ],
        min_length_secs = 1.0,
        max_length_secs = 60*30,
        duration_method = "header",
    )
    print(f"{len(paths)} negative clips after filtering, representing ~{sum(durations)} sec")


def test_negative_features():
    negative_clips, durations = openwakeword.data.filter_audio_paths(
        [
            "data/huggingface/common_voice_17/en/train",
        ],
        min_length_secs = 1.0,
        max_length_secs = 60,
        duration_method = "header",
    )
    print(f"{len(negative_clips)} negative clips after filtering, representing ~{sum(durations)} sec")
    audio_dataset = datasets.Dataset.from_dict({"audio": negative_clips})
    audio_dataset = audio_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    print(f"{audio_dataset.num_rows=} {audio_dataset=}")

    # Get audio embeddings (features) for negative clips and save to .npy file
    # Process files by batch and save to Numpy memory mapped file so that
    # an array larger than the available system memory can be created
    F = openwakeword.utils.AudioFeatures()
    batch_size = 64 # number of files to load, compute features, and write to mmap at a time
    clip_size = 3  # the desired window size (in seconds) for the trained openWakeWord model
    N_total = int(sum(durations)//clip_size) # maximum number of rows in mmap file
    n_feature_cols = F.get_embedding_shape(clip_size)
    print(f"{batch_size=} {clip_size=} {N_total=} {n_feature_cols=}")

    output_file = "data/huggingface/common_voice_17/en/train_features.npy"
    output_array_shape = (N_total, n_feature_cols[0], n_feature_cols[1])
    fp = np.lib.format.open_memmap(output_file, mode='w+', dtype=np.float32, shape=output_array_shape)

    row_counter = 0
    for i in tqdm(np.arange(0, audio_dataset.num_rows, batch_size)):
        # Load data in batches and shape into rectangular array
        wav_data = [(j["array"]*32767).astype(np.int16) for j in audio_dataset[i:i+batch_size]["audio"]]
        print(f"file_wav: {len(wav_data)=} {wav_data[0].shape=}")
        wav_data = openwakeword.data.stack_clips(wav_data, clip_size=16000*clip_size).astype(np.int16)
        print(f"stack_clips: {len(wav_data)=} {wav_data.shape=}")
        
        # Compute features (increase ncpu argument for faster processing)
        features = F.embed_clips(x=wav_data, batch_size=1024, ncpu=8)
        print(f"features: {len(features)=} {features.shape=}")
        
        # Save computed features to mmap array file (stopping once the desired size is reached)
        if row_counter + features.shape[0] > N_total:
            fp[row_counter:min(row_counter+features.shape[0], N_total), :, :] = features[0:N_total - row_counter, :, :]
            fp.flush()
            break
        else:
            fp[row_counter:row_counter+features.shape[0], :, :] = features
            row_counter += features.shape[0]
            fp.flush()

    # Trip empty rows from the mmapped array
    openwakeword.data.trim_mmap(output_file)


def test_positive_features():
    negative_clips, negative_durations = openwakeword.data.filter_audio_paths(
        [
            "data/huggingface/common_voice_17/en/train",
        ],
        min_length_secs = 1.0,
        max_length_secs = 60,
        duration_method = "header",
    )
    print(f"{len(negative_clips)} negative clips after filtering, representing ~{sum(negative_durations)} sec")

    # Get positive example paths, filtering out clips that are too long or too short
    positive_clips, durations = openwakeword.data.filter_audio_paths(
        [
            "data/turn_on_the_office_lights/positive"
        ],
        min_length_secs = 1.0, # minimum clip length in seconds
        max_length_secs = 2.0, # maximum clip length in seconds
        duration_method = "header" # use the file header to calculate duration
    )
    print(f"{len(positive_clips)} positive clips after filtering, representing ~{sum(durations)} sec")

    # Define starting point for each positive clip based on its length, so that each one ends 
    # between 0-200 ms from the end of the total window size chosen for the model.
    # This results in the model being most confident in the prediction right after the
    # end of the wakeword in the audio stream, reducing latency in operation.

    # Get start and end positions for the positive audio in the full window
    sr = 16000
    total_length_seconds = 3 # must be the some window length as that used for the negative examples
    total_length = int(sr*total_length_seconds)

    jitters = (np.random.uniform(0, 0.2, len(positive_clips))*sr).astype(np.int32)
    starts = [total_length - (int(np.ceil(i*sr))+j) for i,j in zip(durations, jitters)]
    ends = [int(i*sr) + j for i, j in zip(durations, starts)]

    # Create generator to mix the positive audio with background audio
    batch_size = 8
    mixing_generator = openwakeword.data.mix_clips_batch(
        foreground_clips = positive_clips,
        background_clips = negative_clips,
        combined_size = total_length,
        batch_size = batch_size,
        snr_low = 5,
        snr_high = 15,
        start_index = starts,
        volume_augmentation=True, # randomly scale the volume of the audio after mixing
    )

    # (Optionally) listen to mixed clips to confirm that the mixing appears correct
    mixed_clips, labels, background_clips = next(mixing_generator)
    AudioPlayer.play_data(mixed_clips[0], samplerate=16000, dtype='float32')

    # Iterate through the mixing generator, computing audio features for positive examples and saving them
    N_total = len(positive_clips) # maximum number of rows in mmap file
    F = openwakeword.utils.AudioFeatures()
    n_feature_cols = F.get_embedding_shape(total_length_seconds)
    print(f"{batch_size=} {total_length_seconds=} {N_total=} {n_feature_cols=}")

    output_file = "data/turn_on_the_office_lights/positive_features.npy"
    output_array_shape = (N_total, n_feature_cols[0], n_feature_cols[1])
    fp = np.lib.format.open_memmap(output_file, mode='w+', dtype=np.float32, shape=output_array_shape)

    row_counter = 0
    for batch in tqdm(mixing_generator, total=N_total//batch_size):
        batch, lbls, background = batch[0], batch[1], batch[2]
        
        # Compute audio features
        features = F.embed_clips(batch, batch_size=256)

        # Save computed features
        fp[row_counter:row_counter+features.shape[0], :, :] = features
        row_counter += features.shape[0]
        fp.flush()
        
        if row_counter >= N_total:
            break

    # Trip empty rows from the mmapped array
    openwakeword.data.trim_mmap(output_file)

