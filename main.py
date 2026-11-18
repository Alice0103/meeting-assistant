"""会议语音助手入口：采集 → 转写 → LLM → 网页展示。"""
import argparse
import logging
import queue
import sys

import uvicorn

from audio import AudioCapture, EnergyVAD, find_input_device, list_input_devices
from events import EventBus

log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="会议语音助手")
    p.add_argument("--list-devices", action="store_true", help="列出输入设备后退出")
    p.add_argument("--device", default=None,
                   help="输入设备名或索引（默认自动找 BlackHole）")
    p.add_argument("--mic", action="store_true", help="使用默认麦克风（调试用）")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-llm", action="store_true", help="只转写不调用 LLM（调试用）")
    p.add_argument("--debug", action="store_true", help="打印调试日志")
    return p.parse_args()


def resolve_device(args) -> int | str | None:
    if args.mic:
        return None
    if args.device:
        try:
            return int(args.device)
        except ValueError:
            return args.device
    idx = find_input_device("BlackHole")
    if idx is None:
        print("未找到 BlackHole 输入设备，用 --list-devices 查看可用设备，"
              "或先安装/启用 BlackHole。也可用 --mic 用麦克风调试。", file=sys.stderr)
        sys.exit(1)
    return idx


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.list_devices:
        print("输入设备：")
        for idx, name, ch, sr in list_input_devices():
            mark = " <- BlackHole" if "blackhole" in name.lower() else ""
            print(f"  [{idx}] {name}  {ch}ch  {sr}Hz{mark}")
        return

    bus = EventBus()
    segment_q: queue.Queue = queue.Queue(maxsize=200)
    text_q: queue.Queue = queue.Queue(maxsize=200)

    device = resolve_device(args)
    vad = EnergyVAD(sr=16000)
    capture = AudioCapture(device, segment_q, vad)

    from transcriber import Transcriber
    transcriber = Transcriber(segment_q, text_q, bus)

    threads = [capture, transcriber]
    if not args.no_llm:
        from assistant import Assistant
        threads.append(Assistant(text_q, bus))

    from web import create_app
    app = create_app(bus, threads)
    print(f"\n打开浏览器访问: http://127.0.0.1:{args.port}\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
