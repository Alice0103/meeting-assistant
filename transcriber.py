"""faster-whisper 转写工作线程。

消费三类任务：
- ("roll", uid, 累计音频, 已覆盖秒数)：说话过程中的滚动预转写，只转新增部分
- (uid, 完整音频, True)：说话结束的完整段，优先复用滚动结果、只补转尾巴
- (uid, 完整音频, False)：20s 强切段（罕见），同样复用滚动结果
"""
import logging
import queue
import threading

from faster_whisper import WhisperModel

from events import EventBus

log = logging.getLogger(__name__)

SR = 16000

INITIAL_PROMPT = "以下是普通话会议内容，请用简体中文转写。"

HALLUCINATION_PATTERNS = (
    "谢谢观看",
    "字幕由",
    "请不吝点赞",
    "订阅我们",
    "一键三连",
    "请用简体中文转写",
)


class Transcriber(threading.Thread):
    def __init__(self, segment_q: queue.Queue, text_q: queue.Queue, bus: EventBus,
                 model_size: str = "small", device: str = "cpu",
                 compute_type: str = "int8", cpu_threads: int = 8):
        super().__init__(name="transcriber", daemon=True)
        self.segment_q = segment_q
        self.text_q = text_q
        self.bus = bus
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self._stop = threading.Event()
        self._model = None
        self._roll: tuple[int, float, str] | None = None
        self._last_done_id = 0
        self._last_final_text = ""

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        log.info("loading whisper model %s (%s/%s)...",
                 self.model_size, self.device, self.compute_type)
        self._model = WhisperModel(
            self.model_size, device=self.device,
            compute_type=self.compute_type, cpu_threads=self.cpu_threads,
        )
        log.info("whisper model ready")
        while not self._stop.is_set():
            try:
                item = self.segment_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            if item[0] == "roll":
                _, uid, audio, covered = item
                if uid <= self._last_done_id:
                    continue
                text = self._transcribe_roll(uid, audio, covered)
                if text:
                    self.bus.publish({"type": "interim", "text": text})
            else:
                uid, audio, is_final = item
                self.bus.publish({"type": "status", "state": "transcribing"})
                text = self._transcribe_finish(uid, audio)
                self._last_done_id = uid
                self.bus.publish({"type": "status", "state": "listening"})
                if text and not self._is_duplicate(text):
                    self._last_final_text = text
                    self.bus.publish({"type": "transcript", "text": text,
                                      "final": is_final})
                    try:
                        self.text_q.put_nowait((text, is_final))
                    except queue.Full:
                        log.warning("text 队列满，丢弃: %s", text[:40])
        log.info("transcriber stopped")

    def _transcribe_roll(self, uid: int, audio, covered: float) -> str:
        """滚动预转写：只转上次覆盖点之后的新增部分。"""
        prev = self._roll if self._roll and self._roll[0] == uid else None
        if prev:
            start = int(prev[1] * SR)
            delta = audio[start:]
        else:
            delta = audio
        if len(delta) < SR // 10:  # 不足 0.1s 没有转写价值
            return prev[2] if prev else ""
        t = self._transcribe(delta)
        joined = (prev[2] + t) if prev else t
        self._roll = (uid, covered, joined)
        return joined

    def _transcribe_finish(self, uid: int, audio) -> str:
        """整段结束：滚动结果已覆盖 95% 以上就直接用，否则只补转尾巴。"""
        n = len(audio)
        roll = self._roll
        text = ""
        if roll and roll[0] == uid:
            covered_samples = int(roll[1] * SR)
            if covered_samples >= n - SR // 2:
                text = roll[2]
            else:
                tail = self._transcribe(audio[covered_samples:])
                text = roll[2] + tail
        else:
            text = self._transcribe(audio)
        if roll and roll[0] == uid:
            self._roll = None
        return text

    def _is_duplicate(self, text: str) -> bool:
        """过滤回声/重复：与上一句几乎相同的内容不再触发回答。"""
        prev = self._last_final_text
        if not prev:
            return False
        if text == prev:
            return True
        if text.startswith(prev) and len(text) - len(prev) < 6:
            return True
        return False

    def _transcribe(self, seg) -> str:
        try:
            segments, _ = self._model.transcribe(
                seg, language="zh", beam_size=1,
                condition_on_previous_text=True,
                initial_prompt=INITIAL_PROMPT,
            )
            text = "".join(s.text for s in segments).strip()
        except Exception:
            log.exception("transcribe failed")
            return ""
        if len(text) < 2:
            return ""
        if any(p in text for p in HALLUCINATION_PATTERNS):
            return ""
        return text
