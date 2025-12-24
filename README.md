# Wan2.2 Video Generation Script

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Automation-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=flat-square)

A robust automation CLI specifically designed for the **Wan2.2 Image-to-Video (I2V)** pipeline. This tool streamlines the generation of high-quality video content by automating interactions with a remote ComfyUI instance (e.g., RunPod). It handles local-to-remote asset uploads, complex multi-segment generation workflows via JSON, and automatic retrieval of the final rendered video.

## 🚀 Key Features

- **Remote Automation**: seamless integration with ComfyUI via `comfy_script`.
- **Smart Asset Management**: Automatically detects if input files are local, uploads them via SCP to the remote server, and downloads the final result.
- **Complex Segmenting**: Supports a `segments-json` argument to chain multiple video generation segments together (e.g., generating frame 1-81, then using the last frame to generate 81-161).
- **LoRA Support**: Granular control over High and Low noise LoRAs with strength configuration.
- **Video Merging**: Automatically merges generated batches into a single H.264 MP4 output.

## 🛠️ Prerequisites

### Client Side

- Python 3.8+
- SSH Access to the compute instance (for file transfer).

### Server Side (ComfyUI)

- **Wan2.2 Models**: `wan2.2_i2v_high_noise_14B_fp16.safetensors`, `wan2.2_i2v_low_noise_14B_fp16.safetensors`.
- **Support Models**: `umt5_xxl_fp16.safetensors` (CLIP), `wan_2.1_vae.safetensors` (VAE).
- **Comfy Script**: The server must be accessible via the ComfyUI API.

## 📦 Installation

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required Python dependency:

   ```bash
   pip install comfy-script
   ```

   _(Note: Standard libraries `argparse`, `json`, `os`, `subprocess` are included in Python)_

## 📖 Usage

### Basic Usage (Single Prompt)

Generate a single video segment from a local image, uploading it to a remote server.

```bash
python run.py \
  --proxy "[https://your-runpod-id-8188.proxy.runpod.net](https://your-runpod-id-8188.proxy.runpod.net)" \
  --input "./assets/my_image.png" \
  --ssh-target "root@123.45.67.89" \
  --ssh-key "~/.ssh/id_rsa" \
  --prompt "A cinematic drone shot of a futuristic city" \
  --length 81
```

### Advanced Usage (JSON Segments)

For complex workflows involving multiple shots or chaining generations.

```bash
python run.py \
  --proxy "[https://your-runpod-id-8188.proxy.runpod.net](https://your-runpod-id-8188.proxy.runpod.net)" \
  --input "./assets/start_frame.png" \
  --ssh-target "root@123.45.67.89" \
  --segments-json "./workflow.json"
```

### JSON Configuration Reference

The `--segments-json` file expects an array of objects. Each object represents a video segment.

| Key                      | Type        | Description                                                                                                       |
| :----------------------- | :---------- | :---------------------------------------------------------------------------------------------------------------- |
| `prompt`                 | `string`    | **Required.** The text prompt for this segment.                                                                   |
| `length`                 | `int`       | Length of the video segment in frames (default: `81`).                                                            |
| `seed`                   | `int`       | Specific seed for reproducibility (default: random).                                                              |
| `lora_high`              | `list[str]` | List of high-noise LoRAs (e.g., `["lora_name:0.8"]`). Overrides previous segment settings.                        |
| `lora_low`               | `list[str]` | List of low-noise LoRAs. Overrides previous segment settings.                                                     |
| `nth_last_frame`         | `int`       | Determines which frame from the end of the batch is selected as the output frame for this segment (default: `1`). |
| **Start Image Controls** |             |                                                                                                                   |
| `start_image`            | `str`       | Path to an explicit start image file.                                                                             |
| `start_from_segment`     | `int`       | Index of a previously generated segment to grab the start frame from.                                             |
| `start_nth_last`         | `int`       | Used with `start_from_segment`: which frame from the end of that segment to use (default: `1`).                   |
| **End Image Controls**   |             |                                                                                                                   |
| `end_image`              | `str`       | Path to an explicit end image file (triggers First-Last frame generation).                                        |
| `use_last_frame_as_end`  | `bool`      | If `true`, uses the last generated frame from the _previous_ loop iteration as the _end_ image for this segment.  |
| `end_from_segment`       | `int`       | Index of a previously generated segment to grab the end frame from.                                               |
| `end_nth_last`           | `int`       | Used with `end_from_segment`: which frame from the end of that segment to use.                                    |

#### Full Example

```json
[
  {
    "prompt": "A cyberpunk city street at night",
    "length": 81,
    "lora_high": ["NeonStyle:0.8"],
    "seed": 123456789
  },
  {
    "prompt": "Camera pans up to the sky",
    "length": 81,
    "start_from_segment": 0,
    "start_nth_last": 1,
    "lora_low": ["MovementHelper:0.5"]
  }
]
```

## ⚙️ Configuration Options

| Argument          | Type   | Default      | Description                                                               |
| :---------------- | :----- | :----------- | :------------------------------------------------------------------------ |
| `--proxy`         | `str`  | **Required** | The HTTP/HTTPS URL to the ComfyUI instance (e.g., RunPod proxy).          |
| `--input`         | `str`  | **Required** | Local path to image (will be uploaded) or remote path if already present. |
| `--ssh-target`    | `str`  | `None`       | SSH connection string `user@host` for file transfer.                      |
| `--ssh-port`      | `str`  | `22`         | SSH port.                                                                 |
| `--ssh-key`       | `str`  | `None`       | Path to private SSH key file.                                             |
| `--output-dir`    | `str`  | `./output`   | Local directory to save the final downloaded video.                       |
| `--segments-json` | `str`  | `None`       | Path to JSON file or JSON string defining the generation timeline.        |
| `--prompt`        | `str`  | `None`       | Single prompt mode (ignored if JSON is provided).                         |
| `--lora-high`     | `list` | `None`       | High noise LoRAs (format: `name` or `name:strength`).                     |
| `--lora-low`      | `list` | `None`       | Low noise LoRAs (format: `name` or `name:strength`).                      |
| `--length`        | `int`  | `81`         | Number of frames for the first instance.                                  |

## 🧩 Workflow Logic

1. **Input Detection**: Checks if `--input` exists locally. If so, uploads to `/workspace/input` on the remote server via SCP.
2. **Model Loading**: Initializes CLIP, VAE, and Wan2.2 High/Low noise UNETs with `comfy_script`.
3. **Generation Loop**: Iterates through the provided CLI args or JSON segments.
   - Handles `start_image` and `end_image` logic (chaining from previous segments).
   - Applies specified LoRAs dynamically per segment.
4. **Merge & Render**: Merges all generated batches and renders a `h264-mp4` video.
5. **Retrieval**: Downloads the final `.mp4` from the remote server to `--output-dir`.

## 📄 License

[MIT License](LICENSE)
