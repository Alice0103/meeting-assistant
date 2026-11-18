"""把 wav 文件播放到系统默认输出（多输出设备会自动分流进 BlackHole）。

用于无人工的全链路测试。注意：不能直接向 BlackHole 输出播放——
BlackHole 在多输出设备聚合中时直接写入会报 -10863。
"""
import sys
import wave

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

wav_path = sys.argv[1]

with wave.open(wav_path, "rb") as w:
    sr = w.getframerate()
    data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

print("playing to default output (多输出设备)")
sd.play(data, sr)
sd.wait()
print("playback done")
