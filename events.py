"""线程安全的事件总线：工作线程发布，asyncio 侧订阅（SSE 广播）。"""
import asyncio
import itertools
import logging
import threading

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self, history_size: int = 500, per_client_size: int = 200):
        self._history: list[dict] = []
        self._history_size = history_size
        self._per_client_size = per_client_size
        self._subscribers: set[asyncio.Queue] = set()
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, event: dict) -> None:
        with self._lock:
            event = {**event, "seq": next(self._seq)}
            self._history.append(event)
            if len(self._history) > self._history_size:
                del self._history[: len(self._history) - self._history_size]
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        for q in list(self._subscribers):
            loop.call_soon_threadsafe(self._put_nowait, q, event)

    def _put_nowait(self, q: asyncio.Queue, event: dict) -> None:
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(event)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._per_client_size)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def history(self) -> list[dict]:
        with self._lock:
            return list(self._history)
