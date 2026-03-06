import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import torch
import time
import pyqtgraph as pg
import pyqtgraph.exporters as exporters
from pyqtgraph.Qt import QtCore
from source.models import SileroVAD
from source.utils import AudioPlayer, AudioFile, FilePlayer, MicPlayer


work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VisualVAD:
    def __init__(self, wav_path: str, save_path: str, samplerate=16000, framesize=512):
        self.save_path = save_path
        self.samplerate = samplerate
        self.framesize = framesize
        self.audiofile = AudioFile(wav_path)
        self.chunks = self.audiofile.get_chunks(
            chunksize=self.framesize,
            samplerate=self.samplerate,
            enable_mono=True,
            dtype='float32',
        )
        self.n_audio = len(self.audiofile.data)
        self.n_frames = int(np.ceil(self.n_audio / self.framesize))
        self.current_frame = 0
        self.total_samples = 0
        self.window_sec = 1.5
        self.window_size = int(self.samplerate * self.window_sec)
        self.wave_buffer = np.zeros(self.window_size, dtype=np.float32)
        self.vad_buffer = np.zeros(self.window_size, dtype=np.float32)
        self.full_waveform = np.zeros(self.n_audio, dtype=np.float32)
        self.vad_results = np.zeros(self.n_frames, dtype=np.float32)
        self.base_x = np.arange(self.window_size) / self.samplerate
        self.model = SileroVAD()
        self.player = AudioPlayer(samplerate=self.samplerate, channels=1)

    def play(self):
        pg.setConfigOptions(antialias=True)
        #pg.setConfigOptions(useOpenGL=True)

        app = pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget(title="Audio + VAD Player")
        self.win.resize(1000, 700)

        self.ax_wave = self.win.addPlot(title="Waveform (Realtime)")
        self.ax_wave.setYRange(-1, 1)
        self.wave_curve = self.ax_wave.plot(self.base_x, self.wave_buffer, pen=pg.mkPen((100, 200, 255), width=1))
        self.wave_curve.setClipToView(True)
        self.wave_curve.setDownsampling(auto=True)

        self.progress_line = pg.InfiniteLine(angle=90, pen='r')
        self.ax_wave.addItem(self.progress_line)

        self.win.nextRow()

        self.ax_vad = self.win.addPlot(title="VAD Probability (Realtime)")
        self.ax_vad.setYRange(0, 1)
        self.vad_curve = self.ax_vad.plot(self.base_x, self.vad_buffer, pen=pg.mkPen('r', width=2))

        self.win.show()
        self.player.start()
    
        interval_ms = int((self.framesize / self.samplerate) * 1000)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(interval_ms)

        pg.exec()

    def _update(self):
        if self.current_frame >= self.n_frames:
            self.timer.stop()
            self.player.stop()
            self._save_full_result()
            return

        start_idx = self.current_frame * self.framesize
        end_idx = min(start_idx + self.framesize, self.n_audio)
        chunk = self.audiofile.data[start_idx:end_idx]
        #chunk = self.chunks[self.current_frame]
        chunk_len = len(chunk)
        self.player.raw_play(chunk)

        prob = float(self.model.process_chunk(chunk, self.samplerate))
        self.full_waveform[start_idx:end_idx] = chunk
        self.vad_results[self.current_frame] = prob

        self.wave_buffer[:-chunk_len] = self.wave_buffer[chunk_len:]
        self.wave_buffer[-chunk_len:] = chunk
        self.vad_buffer[:-chunk_len] = self.vad_buffer[chunk_len:]
        self.vad_buffer[-chunk_len:] = prob

        self.total_samples += chunk_len
        current_time = self.total_samples / self.samplerate
        offset = current_time - self.window_sec

        x_axis = self.base_x + offset
        self.wave_curve.setData(x_axis, self.wave_buffer)
        self.vad_curve.setData(x_axis, self.vad_buffer)
        self.ax_wave.setXRange(offset, current_time)
        self.ax_vad.setXRange(offset, current_time)
        self.progress_line.setValue(current_time)

        print(f"frame[{self.current_frame}] prob={prob:.4f}")
        self.current_frame += 1

    def _save_full_result(self):
        win = pg.GraphicsLayoutWidget(title="Full Result")
        win.resize(1200, 800)

        ax1 = win.addPlot(title="Full Waveform")
        t = np.arange(self.n_audio) / self.samplerate
        ax1.plot(t, self.full_waveform, pen=pg.mkPen((100, 200, 255), width=1))
        ax1.setLabel('bottom', 'Time', 's')
        ax1.setYRange(-1, 1)

        win.nextRow()

        ax2 = win.addPlot(title="VAD Probability")
        t_vad = (np.arange(self.n_frames) * self.framesize) / self.samplerate

        ax2.plot(t_vad, self.vad_results, pen=pg.mkPen('r', width=2))
        ax2.setYRange(0, 1)
        ax2.setLabel('bottom', 'Time', 's')

        win.show()

        exporter = exporters.ImageExporter(win.scene())
        exporter.parameters()['width'] = 2000
        exporter.export(self.save_path)
        print("Saved result image:", self.save_path)


def test_vad():
    v = VisualVAD(f"{work_dir}/data/vad/test.wav", f"{work_dir}/data/vad/vad_result.png")
    v.play()
