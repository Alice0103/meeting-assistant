# 会议语音助手

采集电脑系统音频 → 本地实时转写（faster-whisper small）→ Claude 流式回答/摘要 → 浏览器展示。

## 安装

```bash
conda create -n meeting python=3.13 -y
~/miniconda3/envs/meeting/bin/pip install -r requirements.txt
```

首次运行会自动下载 whisper small 模型（约 460MB）。国内网络直接访问 huggingface.co 会超时，请用镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com ~/miniconda3/envs/meeting/bin/python main.py
```

模型缓存于 `~/.cache/huggingface/`，下载成功后无需再带该变量。

## 运行

```bash
~/miniconda3/envs/meeting/bin/python main.py
# 打开 http://127.0.0.1:8765
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--list-devices` | 列出所有输入设备 |
| `--device 名称或索引` | 指定采集设备（默认自动找 BlackHole） |
| `--mic` | 用默认麦克风（调试链路用） |
| `--no-llm` | 只转写不调用 LLM（调试链路用） |
| `--port N` | 端口，默认 8765 |

## 采集系统音频（BlackHole 路由）

要让助手听见电脑播放的声音（如会议软件、视频）：

1. 打开 **音频 MIDI 设置**（Spotlight 搜 "Audio MIDI Setup"）
2. 左下角 **+** → **创建多输出设备**（Multi-Output Device）
3. 勾选 **BlackHole 2ch** 和你的扬声器/耳机，**采样率必须统一设为 48,000 Hz**（选中多输出设备后把「格式」改成 48000Hz；采样率不匹配会导致 BlackHole 采集报错或静音；BlackHole 子设备开 drift correction）
4. 系统设置 → 声音 → 输出改为这个 **多输出设备**

这样声音同时进扬声器和 BlackHole，助手从 BlackHole 采集。用麦克风测试请给终端授予麦克风权限（系统设置 → 隐私与安全性 → 麦克风）。

## 冒烟测试

```bash
~/miniconda3/envs/meeting/bin/python smoke_test.py --check-env  # 导入 + 设备列表
~/miniconda3/envs/meeting/bin/python smoke_test.py --api        # 代理连通 + 流式
```

`play_wav.py <wav文件>` 把音频播放到系统默认输出（经多输出设备自动分流进 BlackHole），用于自动化测试：先生成测试语音 `say -v Tingting "测试内容" -o /tmp/t.aiff`，转 wav 后播放。注意不能直接向 BlackHole 输出播放（在聚合设备中会报 -10863）。

## 实测延迟（2026-11，M4 Pro）

| 环节 | 实测 |
|---|---|
| 说话过程中 | 每 3 秒滚动预转写，网页灰色实时显示 |
| 说完 → 完整识别 | 静音判定 0.5s 后立即（无需整句重转） |
| 说完 → 回答开始流出 | ~0.6-0.9s（deepseek-chat 首字 0.63s） |
| 一句话 | 作为完整一段转写，AI 基于整句回答 |

## 链路调试顺序

1. `python main.py --mic --no-llm`：对麦克风说话，网页左栏应出现转写
2. 配置好多输出设备后 `python main.py`：播放中文视频，验证转写 + 右栏 AI 流式回答

## 个性化回答

AI 默认直接回答转写中听到的问题；如果问题是关于你本人的（自我介绍、项目经历等），它会基于用户资料以第一人称回答：

```bash
cp profile.example.md profile.md   # 然后编辑填写，重启程序生效
```

## 说明

- LLM 复用环境变量 `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN`（Python SDK 原生支持），模型名 `deepseek-chat`（实测首字延迟 0.63s，比该代理下的 Claude 模型名快近 3 倍；改 `assistant.py` 里的 `MODEL` 常量可切换）
- 转写为本地推理，音频与文本不出本机（LLM 请求除外）
- 调整延迟：`transcriber.py` 的 `beam_size`（改小更快）、`audio.py` 的 VAD 阈值（`enter_rms`/`silence_end_s`）
