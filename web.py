"""FastAPI 应用：SSE 事件流 + 静态页面，lifespan 里启停工作线程。"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from events import EventBus

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app(bus: EventBus, threads: list) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bus.attach_loop(asyncio.get_running_loop())
        for t in threads:
            t.start()
        log.info("pipeline started (%d threads)", len(threads))
        yield
        for t in threads:
            t.stop()
        for t in threads:
            t.join(timeout=5)
        log.info("pipeline stopped")

    app = FastAPI(lifespan=lifespan)

    @app.get("/api/events")
    async def events(request: Request):
        async def gen():
            for ev in bus.history():
                yield _sse(ev)
            q = bus.subscribe()
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15)
                        yield _sse(ev)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
            finally:
                bus.unsubscribe(q)

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/health")
    async def health():
        return {"ok": True, "threads": [t.name for t in threads if t.is_alive()]}

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
