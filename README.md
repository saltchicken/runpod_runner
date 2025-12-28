# RunPod Runner

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Alpha-orange)

**RunPod Runner** is a specialized automation tool designed to orchestrate Wan2.2 Video Generation workflows on RunPod instances. By leveraging `comfy-script`, it bridges local CLI inputs with remote ComfyUI execution, handling complex video tasks such as seamless looping, video splicing, and frame extraction automatically.

This project is optimized for high-end video synthesis using the Wan2.1/2.2 model family (14B parameters) with support for LoRAs and advanced noise scheduling.

## 🚀 Features

- **Remote Execution**: Connects to a RunPod ComfyUI instance via proxy URL to offload heavy GPU computation.
- **Wan2.2 Integration**: Native support for Wan2.2 Image-to-Video (I2V) workflows, including High and Low noise model loading.
- **Smart LoRA Loading**: Dynamically loads single or multiple LoRAs with adjustable strengths via CLI.
- **Advanced Video Handling**:
  - **Concatenation**: Automatically stitches generated segments with input videos using `ffmpeg`.
  - **Looping**: Creates seamless loops by extracting first/last frames and using them as generation targets.
  - **Prepend Mode**: Generates video content leading _up to_ a specific frame in an existing video.
- **Configurable Segments**: Support for JSON-based presets to manage complex prompt/LoRA configurations.

## 🛠️ Tech Stack

- **Language**: Python 3.11+
- **Core Libraries**:
  - `comfy-script`: For programmatic ComfyUI workflow execution.
  - `Pillow`: Image processing.
  - `requests`: HTTP communication with the ComfyUI server.
- **External Tools**: `ffmpeg` (Required for local video processing).
- **Infrastructure**: Docker (RunPod), PyTorch.

## 📋 Prerequisites

1. **Python 3.11** or higher.
2. **FFmpeg**: Must be installed and available in your system `PATH`.
    - _Linux_: `sudo apt install ffmpeg`
    - _macOS_: `brew install ffmpeg`
    - _Windows_: Download and add to PATH.
3. **RunPod Instance**: A running GPU pod with ComfyUI and the required custom nodes (specifically `ComfyScript` and `Wan2.2` models).

## 📦 Installation

### Local Client Setup

Clone the repository and install the package in editable mode:

```bash
git clone [https://github.com/yourusername/runpod-runner.git](https://github.com/yourusername/runpod-runner.git)
cd runpod-runner
pip install -e .
```

### Server Setup (Docker)

If you are building the RunPod image yourself, use the provided `Dockerfile`. It creates an environment based on `runpod/pytorch:1.0.3` with ComfyUI and necessary nodes pre-installed.

```bash
docker build -t runpod-runner:latest .
```

## 💻 Usage

The tool is executed via the `runpod_runner` command. You must provide a proxy URL to your RunPod instance.

### Basic Image-to-Video

Generate a video from a local image:

```bash
python -m runpod_runner \
  --proxy "[https://your-pod-id-3000.proxy.runpod.net](https://your-pod-id-3000.proxy.runpod.net)" \
  --input "./assets/input_image.png" \
  --prompt "A cyberpunk city in rain, cinematic lighting" \
  --length 81
```

### Video Loop Generation

Take an existing video, extract the last frame, and generate a segment that seamlessly connects back to the start (Append mode):

```bash
python -m runpod_runner \
  --proxy "[https://your-pod-id-3000.proxy.runpod.net](https://your-pod-id-3000.proxy.runpod.net)" \
  --input "./assets/video.mp4" \
  --loop \
  --prompt "ocean waves transforming into digital data"
```

### Prepend Mode

Generate video content that _precedes_ the input video (Reverse flow logic):

```bash
python -m runpod_runner \
  --proxy "[https://your-pod-id-3000.proxy.runpod.net](https://your-pod-id-3000.proxy.runpod.net)" \
  --input "./assets/video.mp4" \
  --prepend \
  --video-splice-time 0.0 \
  --prompt "A calm sea before the storm"
```

### Using LoRAs

Apply specific LoRAs to the generation. Format is `name:strength` (strength defaults to 1.0 if omitted).

```bash
python -m runpod_runner \
  --proxy "[https://your-pod-id-3000.proxy.runpod.net](https://your-pod-id-3000.proxy.runpod.net)" \
  --input "./image.png" \
  --prompt "Anime style character" \
  --lora-high "anime_style.safetensors:0.8" \
  --lora-low "detail_enhancer.safetensors:0.5"
```

## ⚙️ Configuration & Arguments

| Argument              | Description                                               | Default         |
| :-------------------- | :-------------------------------------------------------- | :-------------- |
| `--proxy`             | **Required.** RunPod proxy URL (port 3000 usually).       | -               |
| `--input`             | Path to local image or MP4 file.                          | Env `INPUT_DIR` |
| `--prompt`            | Text prompt for generation.                               | -               |
| `--segment`           | Path to a JSON config file or raw JSON string.            | -               |
| `--output-dir`        | Directory to save results.                                | `./output`      |
| `--length`            | Number of frames to generate.                             | `81`            |
| `--seed`              | Seed for generation (random if unset).                    | None            |
| `--prepend`           | Generates video leading up to input video.                | False           |
| `--loop`              | Creates a seamless loop (connects start/end).             | False           |
| `--video-splice-time` | Timestamp (sec) to cut input video for splicing.          | None            |
| `--video-target-time` | Timestamp (sec) to extract target frame from input video. | None            |

### Environment Variables

You can use a `.env` file to set default directories:

```ini
INPUT_DIR=./my_inputs
OUTPUT_DIR=./my_outputs
```

## 📄 JSON Segment Configuration

Instead of passing complex CLI arguments, you can use a JSON file with the `--segment` flag:

```json
{
  "prompt": "A futuristic robot dancing",
  "lora_high": ["robot_lora.safetensors:1.0"],
  "lora_low": ["lighting_lora.safetensors:0.6"],
  "start_image": "/optional/override/start.png",
  "end_image": "/optional/override/end.png"
}
```

## 🏗️ Project Structure

```text
.
├── Dockerfile             # RunPod image definition
├── pyproject.toml         # Python dependencies and metadata
├── scripts/               # Server-side startup scripts
└── src/
    └── runpod_runner/
        ├── __main__.py    # CLI entry point and logic
        ├── wan_video.py   # ComfyUI automation wrapper
        └── utils.py       # Helper functions (logging, etc)
```
