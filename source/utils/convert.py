import librosa
import numpy as np


class AudioConvert:
    """
    Audio format conversion utilities.

    Internal canonical format:
        uint8 in range [0, 255], center=128
        uint16 in range [-32767, 32767]
        float32 in range [-1.0, 1.0]
    """

    @staticmethod
    def to_float32(audio: np.ndarray) -> np.ndarray:
        """
        Convert PCM audio to float32 [-1.0, 1.0]

        Supported input:
            uint8 / int16 / int32 / float32 / float64
        """
        if not isinstance(audio, np.ndarray):
            raise TypeError("audio must be a numpy array")

        if np.issubdtype(audio.dtype, np.floating):
            return np.clip(audio.astype(np.float32), -1.0, 1.0)

        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            if audio.dtype == np.uint8:
                # uint8 PCM: [0,255] with zero at 128
                # Negative half-axis: 128 values (0 → 127)
                # Positive half-axis: 127 values ​​(129 → 255)
                audio = (audio.astype(np.float32) - 128.0) / 128.0
            else:
                # signed PCM (int16, int32)
                audio = audio.astype(np.float32)
                audio /= max(abs(info.min), info.max)
            return np.clip(audio, -1.0, 1.0)

        raise TypeError(f"Unsupported audio dtype: {audio.dtype}")

    @staticmethod
    def to_int16(audio: np.ndarray) -> np.ndarray:
        """
        Convert PCM audio to int16 [-32767, 32767]

        Supported input:
            uint8 / int16 / int32 / float32 / float64
        """
        audio_f32 = AudioConvert.to_float32(audio)
        return (audio_f32 * 32767.0).astype(np.int16)

    @staticmethod
    def to_uint8(audio: np.ndarray) -> np.ndarray:
        """
        Convert PCM audio to uint8 [0,255] with zero at 128
        """
        audio_f32 = AudioConvert.to_float32(audio)

        # float [-1,1] -> uint8 PCM
        audio_u8 = audio_f32 * 128.0 + 128.0
        return np.clip(audio_u8, 0, 255).astype(np.uint8)

    @staticmethod
    def convert(audio: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
        """
        Generic conversion entry.
        """
        if target_dtype == np.uint8:
            return AudioConvert.to_uint8(audio)

        if target_dtype == np.int16:
            return AudioConvert.to_int16(audio)

        if target_dtype == np.float32:
            return AudioConvert.to_float32(audio)

        raise ValueError(f"Unsupported target dtype: {target_dtype}")

    @staticmethod
    def transform(audio: np.ndarray, *, orig_sr: int, target_sr: int, enable_mono: bool) -> np.ndarray:
        """
        Canonical transform:
            → mono (optional)
            → resample
            → float32 [-1,1]
        """
        audio = AudioConvert.to_float32(audio)

        if enable_mono and audio.ndim > 1:
            audio = librosa.to_mono(audio)

        if orig_sr != target_sr:
            audio = librosa.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=target_sr,
            )
        return audio

    @staticmethod
    def mixing(
        primary: np.ndarray,
        primary_sr: int,
        *,
        secondary: np.ndarray,
        secondary_sr: int,
        secondary_gain: float,
        output_sr: int,
        start_sec: float,
        duration_sec: float | None = None,
        loop_secondary: bool = False,
        enable_mono: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        General audio mixing function with fixed duration and optional looping.

        Args:
            primary: main audio array, shape (N,) or (C, N)
            primary_sr: sample rate of primary audio
            secondary: audio to mix
            secondary_sr: sample rate of secondary audio
            secondary_gain: gain for secondary
            output_sr: output sample rate
            start_sec: start time in primary to mix in seconds
            duration_sec: max mixing duration (None -> till secondary length)
            loop_secondary: if True, secondary will loop to fill duration
            enable_mono: convert both audios to mono

        Returns:
            Mixed audio (float32, same shape as primary)
        """
        primary = AudioConvert.transform(primary, orig_sr=primary_sr, target_sr=output_sr, enable_mono=enable_mono)
        secondary = AudioConvert.transform(secondary, orig_sr=secondary_sr, target_sr=output_sr, enable_mono=enable_mono)

        # Determine mixing region
        start_sample = int(start_sec * output_sr)
        if duration_sec is None:
            duration_samples = len(secondary)
        else:
            duration_samples = min(int(duration_sec * output_sr), len(primary) - start_sample)

        if duration_samples <= 0 or start_sample >= len(primary):
            return primary  # nothing to mix

        # Trim or loop secondary to match duration
        secondary_section = secondary
        sec_len = len(secondary)
        if sec_len == 0:
            secondary_section = np.zeros(duration_samples, dtype=np.float32)
        elif sec_len < duration_samples:
            if loop_secondary:
                repeats = int(np.ceil(duration_samples / sec_len))
                secondary_section = np.tile(secondary, repeats)[:duration_samples]
            else:
                # pad with zeros
                secondary_section = np.pad(secondary, (0, duration_samples - sec_len), mode='constant')
        else:
            secondary_section = secondary[:duration_samples]

        # Handle channel mismatch (support arbitrary multi-channel)
        if primary.ndim != secondary_section.ndim or (
            primary.ndim == 2 and secondary_section.ndim == 2 and primary.shape[0] != secondary_section.shape[0]
        ):
            if secondary_section.ndim == 1:
                # secondary is mono, primary multichannel: copied to each channel
                secondary_section = np.tile(secondary_section, (primary.shape[0], 1))
            elif primary.ndim == 1:
                # primary is mono, and the secondary is multi-channel: take the first channel
                primary = np.tile(primary, (secondary_section.shape[0], 1))
            else:
                # Both sides are multi-channel with different channels: the loop secondary channel matches the primary
                sec_ch, sec_len = secondary_section.shape
                prim_ch, prim_len = primary.shape
                repeats = int(np.ceil(prim_ch / sec_ch))
                secondary_section = np.tile(secondary_section, (repeats, 1))[:prim_ch, :sec_len]

        # Mix and normalize/clip (always on)
        mixed_section = primary[..., start_sample:start_sample + duration_samples] + secondary_gain * secondary_section
        peak = np.max(np.abs(mixed_section))
        if peak > 1.0:
            mixed_section /= peak
        mixed_section = np.clip(mixed_section, -1.0, 1.0)
        primary[..., start_sample:start_sample + duration_samples] = mixed_section
        return primary
