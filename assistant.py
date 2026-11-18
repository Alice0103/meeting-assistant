"""LLM 回答工作线程：消费转写文本，流式调用 Claude 并广播回答。"""
import logging
import queue
import threading
import time
from pathlib import Path

import anthropic

from events import EventBus

log = logging.getLogger(__name__)

PROFILE_PATH = Path(__file__).parent / "profile.md"


def _load_profile() -> str:
    try:
        return PROFILE_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "（暂无用户资料，请创建 profile.md）"

SYSTEM_PROMPT = """你是一个实时语音助手，正在旁听对话并实时收到语音转写。
转写文本按时间顺序排列：「[未完成片段]」是对方说话过程中的碎片（可能被截断、有错别字），「[完整句]」是对方刚说完的完整内容。

你的任务：
1. 如果对方提出了问题，直接给出这个问题的答案。不要复述问题、不要总结，直接回答。
2. 如果问题是关于用户本人的（自我介绍、个人经历、项目、技能、职业规划等），必须基于下方「用户资料」以用户第一人称回答。不得编造资料中没有的经历或数据；资料不足时按最保守、最合理的方式回答。
3. 只有当对方没有提问、只是在陈述或讨论时，才输出要点总结。
4. 用简体中文回答，口语化、自然、简洁。不要开场白，不要「回答如下」之类的套话。

=== 用户资料 ===
{profile}"""

MODEL = "deepseek-chat"
MAX_TOKENS = 2048
HISTORY_MESSAGES = 8
BACKLOG_LIMIT = 30


class Assistant(threading.Thread):
    def __init__(self, text_q: queue.Queue, bus: EventBus,
                 model: str = MODEL, max_tokens: int = MAX_TOKENS):
        super().__init__(name="assistant", daemon=True)
        self.text_q = text_q
        self.bus = bus
        self.model = model
        self.max_tokens = max_tokens
        self._stop = threading.Event()
        self._client = None
        self._history: list[dict] = []
        self._system = SYSTEM_PROMPT.format(profile=_load_profile())

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        self._client = anthropic.Anthropic()
        log.info("assistant ready (model=%s)", self.model)
        while not self._stop.is_set():
            try:
                first = self.text_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if first is None:
                break
            batch, had_partials = self._drain_batch(first)
            if not batch.strip():
                continue
            self._answer(batch, had_partials)

    def _drain_batch(self, first: tuple) -> tuple[str, bool]:
        """攒齐一句话（直到 final 段）再合并为一条 user 消息。

        返回 (合并文本, 是否包含未完成碎片)。
        """
        texts = [first[0]]
        final = first[1]
        had_partials = not final
        if not final:
            deadline = time.monotonic() + 10.0
            while not final:
                try:
                    text, is_final = self.text_q.get(timeout=0.2)
                except queue.Empty:
                    if time.monotonic() > deadline:
                        break
                    continue
                texts.append(text)
                final = is_final
        if self.text_q.qsize() > BACKLOG_LIMIT:
            skipped = 0
            while self.text_q.qsize() > 0:
                try:
                    self.text_q.get_nowait()
                    skipped += 1
                except queue.Empty:
                    break
            texts.append(f"[已跳过 {skipped} 条转写]")
        lines = [t for t in texts if t]
        if had_partials and len(lines) > 1:
            body = "\n".join(f"[未完成片段] {t}" for t in lines[:-1])
            body += f"\n[完整句] {lines[-1]}"
        else:
            body = "\n".join(lines)
        return body, had_partials

    def _answer(self, text: str, had_partials: bool = False) -> None:
        if self._history and self._history[-1]["role"] == "user":
            self._history[-1]["content"] += "\n" + text
        else:
            self._history.append({"role": "user", "content": text})
        self._history = self._history[-HISTORY_MESSAGES:]
        if self._history[0]["role"] == "assistant":
            self._history = self._history[1:]

        messages = self._with_cache_control([dict(m) for m in self._history])
        self.bus.publish({"type": "status", "state": "thinking"})
        self.bus.publish({"type": "answer_start"})

        partial: list[str] = []
        try:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "disabled"},
                system=[{"type": "text", "text": self._system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=messages,
            ) as stream:
                for delta in stream.text_stream:
                    partial.append(delta)
                    self.bus.publish({"type": "answer_delta", "text": delta})
                final = stream.get_final_message()
                usage = final.usage
                log.info("usage: in=%s out=%s cache_read=%s",
                         usage.input_tokens, usage.output_tokens,
                         getattr(usage, "cache_read_input_tokens", 0))
        except Exception as e:
            log.exception("llm stream failed")
            note = f"\n[回答中断: {type(e).__name__}]"
            self.bus.publish({"type": "answer_delta", "text": note})
            partial.append(note)

        answer = "".join(partial).strip()
        if answer:
            self._history.append({"role": "assistant", "content": answer})
        else:
            # 无任何产出：撤销 user 消息，让下一条转写与之合并
            if self._history and self._history[-1]["role"] == "user":
                self._history.pop()
        self.bus.publish({"type": "status", "state": "listening"})
        self.bus.publish({"type": "answer_done"})

    @staticmethod
    def _with_cache_control(messages: list[dict]) -> list[dict]:
        last = messages[-1]
        content = last["content"]
        blocks = content if isinstance(content, list) else [
            {"type": "text", "text": content}]
        blocks = [dict(b) for b in blocks]
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
        last["content"] = blocks
        return messages
