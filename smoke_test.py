"""冒烟测试：--check-env 验证导入与设备；--api 验证代理连通与流式。"""
import argparse
import sys


def check_env():
    import sounddevice as sd
    import numpy
    import faster_whisper
    import anthropic
    import scipy
    print(f"numpy {numpy.__version__}, scipy {scipy.__version__}, "
          f"faster_whisper {faster_whisper.__version__}, "
          f"anthropic {anthropic.__version__}")
    print("输入设备：")
    found = False
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            mark = " <- BlackHole" if "blackhole" in dev["name"].lower() else ""
            if mark:
                found = True
            print(f"  [{i}] {dev['name']}  {dev['max_input_channels']}ch  "
                  f"{int(dev['default_samplerate'])}Hz{mark}")
    if not found:
        print("警告：未发现 BlackHole 输入设备（用 --mic 模式调试时无影响）")
    print("check-env OK")


def api_test():
    import anthropic
    client = anthropic.Anthropic()
    print("连接代理并流式请求一句中文…")
    try:
        with client.messages.stream(
            model="claude-opus-4-7",
            max_tokens=512,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": "请用一句话介绍你自己"}],
        ) as stream:
            for delta in stream.text_stream:
                print(delta, end="", flush=True)
            final = stream.get_final_message()
        print()
        u = final.usage
        print(f"usage: input={u.input_tokens} output={u.output_tokens} "
              f"cache_read={getattr(u, 'cache_read_input_tokens', 0)}")
        print("api-test OK")
    except Exception as e:
        print(f"api-test FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check-env", action="store_true")
    p.add_argument("--api", action="store_true")
    args = p.parse_args()
    if args.check_env:
        check_env()
    if args.api:
        api_test()


if __name__ == "__main__":
    main()
