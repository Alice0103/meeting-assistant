"""音频采集：sounddevice 从指定输入设备读流 + 能量 VAD 切段。

说话过程中每 3 秒发一次「滚动预转写」任务，让转写线程在说话时就
把前面的话转好；说完后只需补转最后的尾巴，实现说完即出。
"""
import logging
import queue
import threading

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

ROLL_INTERVAL_S = 3.0


def list_input_devices() -> list[tuple[int, str, int, int]]:
    """返回 [(index, name, max_input_channels, default_samplerate)]。"""
    out = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            out.append((i, dev["name"], dev["max_input_channels"], int(dev["default_samplerate"])))
    return out


def find_input_device(pattern: str = "BlackHole") -> int | None:
    """按名称子串找输入设备，优先 2 通道。找不到返回 None。"""
    matches = [d for d in list_input_devices() if pattern.lower() in d[1].lower()]
    if not matches:
        return None
    matches.sort(key=lambda d: d[2] == 2, reverse=True)
    return matches[0][0]


class EnergyVAD:
    """RMS 能量 + 双向迟滞静音检测。

    输出 (话语编号, 音频段, 是否完整结束) 三元组；
    说话过程中可通过 inprogress() 拿到进行中的累计音频做预转写。
    """

    def __init__(
        self,
        sr: int = 16000,
        frame_s: float = 0.02,
        enter_rms: float = 0.008,
        stay_rms: float = 0.003,
        silence_end_s: float = 0.5,
        max_segment_s: float = 20.0,
        min_speech_s: float = 0.3,
        pre_roll_s: float = 0.3,
    ):
        self.sr = sr
        self.frame_s = frame_s
        self.enter_rms = enter_rms
        self.stay_rms = stay_rms
        self.silence_frames = int(round(silence_end_s / frame_s))
        self.max_frames = int(round(max_segment_s / frame_s))
        self.min_frames = int(round(min_speech_s / frame_s))
        self.ring_max = int(round(pre_roll_s / frame_s))

        self._ring: list[np.ndarray] = []
        self._segment: list[np.ndarray] = []
        self._in_speech = False
        self._silent = 0
        self._utterance_id = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def inprogress(self) -> tuple[int, np.ndarray] | None:
        """进行中话语的 (编号, 累计音频)。未在说话返回 None。"""
        if not self._in_speech or not self._segment:
            return None
        return self._utterance_id, np.concatenate(self._segment)

    def push(self, frame: np.ndarray) -> list[tuple[int, np.ndarray, bool]]:
        out: list[tuple[int, np.ndarray, bool]] = []
        rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
        if not self._in_speech:
            self._ring.append(frame)
            if len(self._ring) > self.ring_max:
                del self._ring[: len(self._ring) - self.ring_max]
            if rms >= self.enter_rms:
                self._in_speech = True
                self._utterance_id += 1
                self._segment = list(self._ring)
                self._ring = []
                self._silent = 0
        else:
            self._segment.append(frame)
            if rms < self.stay_rms:
                self._silent += 1
            else:
                self._silent = 0
            if self._silent >= self.silence_frames:
                out.extend(self._finish(keep_speech=False))
            elif len(self._segment) >= self.max_frames:
                out.extend(self._finish(keep_speech=True))
        return out

    def _finish(self, keep_speech: bool) -> list[tuple[int, np.ndarray, bool]]:
        out = []
        seg = np.concatenate(self._segment)
        uid = self._utterance_id
        if len(seg) >= self.min_frames:
            out.append((uid, seg, not keep_speech))
        self._segment = []
        self._silent = 0
        self._in_speech = keep_speech
        if keep_speech:
            # 强切后仍在说话，后续内容算新的一段
            self._utterance_id += 1
        return out


class AudioCapture(threading.Thread):
    """阻塞读输入设备，逐帧喂 VAD，完成段放入 segment_q。"""

    def __init__(self, device, segment_q: queue.Queue, vad: EnergyVAD,
                 samplerate: int = 16000, blocksize: int = 320, fallback_sr: int = 48000):
        super().__init__(name="audio-capture", daemon=True)
        self.device = device
        self.segment_q = segment_q
        self.vad = vad
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.fallback_sr = fallback_sr
        self._stop = threading.Event()
        self._roll_uid = 0
        self._roll_covered = 0.0

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        from scipy.signal import resample_poly

        stream, sr = self._open()
        silent_frames = 0
        reopen_after = int(round(15 * self.samplerate / self.blocksize))
        frames_since_roll = 0
        try:
            while not self._stop.is_set():
                data, _ = stream.read(self.blocksize * sr // self.samplerate)
                if data.ndim > 1:
                    data = data[:, 0]
                if sr != self.samplerate:
                    data = resample_poly(data, self.samplerate, sr).astype(np.float32)
                rms = float(np.sqrt(np.mean(data**2) + 1e-12))
                if rms < 0.0005:
                    silent_frames += 1
                else:
                    silent_frames = 0
                if silent_frames >= reopen_after:
                    # 聚合设备（多输出设备）空闲再激活后，旧采集流可能永远收不到
                    # 音频，静音超过 15s 就重开流恢复
                    log.info("15s 静音，重开采集流")
                    stream.close()
                    stream, sr = self._open()
                    silent_frames = 0
                    continue
                for item in self.vad.push(data):
                    try:
                        self.segment_q.put_nowait(item)
                    except queue.Full:
                        log.warning("segment 队列满，丢弃一段")
                frames_since_roll += 1
                if frames_since_roll >= 25:  # 每 0.5s 检查一次滚动预转写
                    frames_since_roll = 0
                    self._maybe_roll()
        finally:
            stream.close()
        log.info("capture stopped")

    def _maybe_roll(self) -> None:
        """说话过程中每 3 秒发一次预转写任务。"""
        prog = self.vad.inprogress()
        if prog is None:
            return
        uid, audio = prog
        covered = len(audio) / self.vad.sr
        if uid != self._roll_uid:
            self._roll_uid = uid
            self._roll_covered = 0.0
        if covered - self._roll_covered >= ROLL_INTERVAL_S:
            try:
                self.segment_q.put_nowait(("roll", uid, audio, covered))
            except queue.Full:
                log.warning("roll 队列满，跳过")
            self._roll_covered = covered

    def _open(self):
        try:
            stream = sd.InputStream(
                device=self.device, samplerate=self.samplerate, channels=1,
                dtype="float32", blocksize=self.blocksize, latency="high",
            )
            sr = self.samplerate
            log.info("capturing %s at %d Hz", self.device, sr)
        except Exception:
            stream = sd.InputStream(
                device=self.device, samplerate=self.fallback_sr, channels=1,
                dtype="float32", blocksize=self.blocksize * self.fallback_sr // self.samplerate,
                latency="high",
            )
            sr = self.fallback_sr
            log.info("16k 打开失败，回退 %d Hz 采集", sr)
        stream.start()
        return stream, sr
