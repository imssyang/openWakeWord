import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import torch
import time
from source.models import SileroVAD
from source.utils import AudioPlayer, AudioFile, FilePlayer, MicPlayer


work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VisualVAD:
    def __init__(self, path: str):
        self.samplerate = 16000
        self.framesize = 512
        self.audiofile = AudioFile(path)
        self.n_audio = len(self.audiofile.data)
        self.n_frames = self.n_audio // self.framesize
        self.model = SileroVAD()
        self.player = AudioPlayer(samplerate=self.samplerate, channels=1)

    def play(self):
        full_waveform = np.zeros(self.n_audio, dtype=np.float32)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        line_wave, = ax1.plot(np.arange(self.n_audio), full_waveform)
        ax1.set_ylim(-1, 1)
        ax1.set_xlim(0, self.n_audio)
        ax1.set_title("Full Audio Waveform (Real-time Playback)")
        ax1.set_xlabel("Samples")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        progress_line = ax1.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        line_vad, = ax2.plot([], [], 'r', linewidth=2)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, self.n_frames)
        ax2.set_xlabel("Frame Index")
        ax2.set_ylabel("Probability")
        ax2.set_title("VAD Probability")
        ax2.grid(True, alpha=0.3)

        vad_results = []
        current_frame = 0
        chunks = self.audiofile.get_chunks(
            chunksize=self.framesize,
            samplerate=self.samplerate,
            enable_mono=True,
            dtype='float32',
        )

        self.player.start()

        def update(frame):
            nonlocal current_frame, vad_results
            if current_frame >= self.n_frames:
                anim.event_source.stop()
                return line_wave, line_vad

            chunk = chunks[current_frame]
            self.player.raw_play(chunk)

            prob = self.model.process_chunk(chunk, self.samplerate)
            vad_results.append(prob)


            start_idx = current_frame * self.framesize
            end_idx = start_idx + len(chunk)
            full_waveform[start_idx:end_idx] = chunk
            line_wave.set_ydata(full_waveform)
            line_vad.set_data(range(len(vad_results)), vad_results)

            print(f"chunk[{current_frame}] {prob=}")
            current_frame += 1
            return line_wave, line_vad

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(chunks),
            interval=int((self.framesize / self.samplerate) * 1000),
            blit=True,
            repeat=False,
        )
        plt.tight_layout() 
        anim.save(f"{work_dir}/tests/test_audio_vad.mp4", writer='ffmpeg')
        plt.savefig(f"{work_dir}/tests/test_audio_vad.png")
        plt.close(fig)

        self.player.stop()


def test_vad():
    v = VisualVAD(f"{work_dir}/data/vad/test.wav")
    v.play()
