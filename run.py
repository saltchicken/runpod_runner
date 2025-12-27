import argparse
import json
import os
import random
import io
import requests
import subprocess
import shutil
from datetime import datetime
from PIL import Image as PILImage
from comfy_script.runtime import client, load, Workflow
import comfy_script.runtime.util as util


def upload_to_comfy(pil_image, filename="script_input.png"):
    """
    Uploads a PIL image to the ComfyUI server's input directory via HTTP.
    Returns: The filename string expected by LoadImage.
    """
    byte_stream = io.BytesIO()
    pil_image.save(byte_stream, format="PNG")
    byte_stream.seek(0)

    # This ensures it works with your RunPod/Remote setup automatically
    base_url = client.client.base_url  # e.g., "http://127.0.0.1:8188/"
    if not base_url.endswith("/"):
        base_url += "/"
    target_url = f"{base_url}upload/image"

    # We set overwrite=true so you can reuse this filename for every run
    files = {"image": (filename, byte_stream, "image/png")}
    data = {"overwrite": "true"}

    response = requests.post(target_url, files=files, data=data)
    response.raise_for_status()  # Stop if upload fails

    # Return the filename only (which is what LoadImage accepts)
    return filename


def generate_video(
    proxy,
    input_path,
    segment_json=None,
    prompt=None,
    lora_high=None,
    lora_low=None,
    length=81,
):
    """
    Connects to ComfyUI, generates a single video segment, and returns the list of PIL frames.
    """

    load(proxy)

    import comfy_script.runtime.nodes as nodes

    def setup_models():
        clip = nodes.CLIPLoader("umt5_xxl_fp16.safetensors", "wan", "default")
        vae = nodes.VAELoader("wan_2.1_vae.safetensors")

        WAN_HIGH_NOISE_LIGHTNING_STRENGTH = 3.0
        WAN_LOW_NOISE_LIGHTNING_STRENGTH = 1.5

        wan_high_noise_model = nodes.UNETLoader(
            "Wan2.2/wan2.2_i2v_high_noise_14B_fp16.safetensors", "default"
        )
        wan_high_noise_model = nodes.LoraLoaderModelOnly(
            wan_high_noise_model,
            "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
            WAN_HIGH_NOISE_LIGHTNING_STRENGTH,
        )

        wan_low_noise_model = nodes.UNETLoader(
            "Wan2.2/wan2.2_i2v_low_noise_14B_fp16.safetensors", "default"
        )
        wan_low_noise_model = nodes.LoraLoaderModelOnly(
            wan_low_noise_model,
            "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
            WAN_LOW_NOISE_LIGHTNING_STRENGTH,
        )

        return clip, vae, wan_high_noise_model, wan_low_noise_model

    def load_loras(model, loras):
        if isinstance(loras, str):
            loras = [loras]

        if not loras:
            return nodes.ModelSamplingSD3(model, 5)

        def parse_lora(lora_str):
            name = lora_str
            strength = 1.0
            if ":" in lora_str:
                parts = lora_str.rsplit(":", 1)
                try:
                    strength = float(parts[1])
                    name = parts[0]
                except ValueError:
                    pass
            return name, strength

        name, strength = parse_lora(loras[0])
        lora_model = nodes.LoraLoaderModelOnly(model, name, strength)

        if len(loras) >= 2:
            name, strength = parse_lora(loras[1])
            lora_model = nodes.LoraLoaderModelOnly(lora_model, name, strength)

        if len(loras) > 2:
            print("Note: Only first two LoRAs loaded.")

        lora_model = nodes.ModelSamplingSD3(lora_model, 5)
        return lora_model

    def wan_frame_to_video(
        model_high,
        model_low,
        clip,
        vae,
        start_image,
        prompt,
        loras_high,
        loras_low,
        end_image=None,
        length=81,
        seed=None,
    ):
        steps = nodes.PrimitiveInt(8)
        cfg_high = nodes.PrimitiveFloat(1)
        cfg_low = nodes.PrimitiveFloat(1)
        start_low_at_step = nodes.PrimitiveInt(4)

        if seed is None:
            seed = random.randint(0, 0xFFFFFFFFFFFFFF)

        empty_negative_conditioning = nodes.CLIPTextEncode("", clip)
        conditioning = nodes.CLIPTextEncode(prompt, clip)

        print(f"Generating segment with Prompt: '{prompt}'")
        print(f"High Noise LoRAs: {loras_high if loras_high else 'None (Base Model)'}")
        print(f"Seed: {seed}")

        lora_model_high = load_loras(model_high, loras_high)
        lora_model_low = load_loras(model_low, loras_low)

        width, height, _ = nodes.GetImageSize(start_image)

        if end_image is not None:
            positive, negative, latent = nodes.WanFirstLastFrameToVideo(
                conditioning,
                empty_negative_conditioning,
                vae,
                width,
                height,
                length,
                1,
                None,
                None,
                start_image,
                end_image,
            )
        else:
            positive, negative, latent = nodes.WanImageToVideo(
                conditioning,
                empty_negative_conditioning,
                vae,
                width,
                height,
                length,
                1,
                None,
                start_image,
            )

        latent = nodes.KSamplerAdvanced(
            lora_model_high,
            "enable",
            seed,
            steps,
            cfg_high,
            "euler",
            "simple",
            positive,
            negative,
            latent,
            0,
            start_low_at_step,
            "enable",
        )
        latent = nodes.KSamplerAdvanced(
            lora_model_low,
            "disable",
            0,
            steps,
            cfg_low,
            "euler",
            "simple",
            positive,
            negative,
            latent,
            start_low_at_step,
            10000,
            "disable",
        )

        segment1 = nodes.VAEDecode(latent, vae)

        # Returning the decoded frames directly.
        return segment1

    # --- Main Workflow Execution ---
    with Workflow() as wf:
        clip, vae, wan_high_noise_model, wan_low_noise_model = setup_models()

        input_image, _ = nodes.LoadImage(input_path)

        # Now expects a single configuration (dict) or falls back to CLI args.
        config = {}

        if segment_json:
            try:
                loaded_json = None
                if os.path.isfile(segment_json):
                    with open(segment_json, "r") as f:
                        loaded_json = json.load(f)
                else:
                    loaded_json = json.loads(segment_json)

                if isinstance(loaded_json, list):
                    if len(loaded_json) > 0:
                        config = loaded_json[0]
                elif isinstance(loaded_json, dict):
                    config = loaded_json
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                exit(1)

        final_prompt = config.get("prompt", prompt)
        final_lora_high = config.get("lora_high", lora_high)
        final_lora_low = config.get("lora_low", lora_low)
        final_length = config.get("length", length)
        final_seed = config.get("seed", None)

        # We already loaded CLI input as `input_image` above.
        start_image_node = input_image
        if "start_image" in config:
            print(f"Loading explicit start image from JSON: {config['start_image']}")
            start_image_node, _ = nodes.LoadImage(config["start_image"])

        end_image_node = None
        if "end_image" in config:
            print(f"Loading explicit end image: {config['end_image']}")
            end_image_node, _ = nodes.LoadImage(config["end_image"])

        if not final_prompt:
            print("Error: No prompt provided in JSON or CLI.")
            return []

        video_batch = wan_frame_to_video(
            wan_high_noise_model,
            wan_low_noise_model,
            clip,
            vae,
            start_image=start_image_node,
            prompt=final_prompt,
            loras_high=final_lora_high,
            loras_low=final_lora_low,
            end_image=end_image_node,
            length=final_length,
            seed=final_seed,
        )

        print("\n--- Retrieving Frames from Server ---")
        all_video_frames = util.get_images(video_batch)

        print(f"Done! {len(all_video_frames)} frames returned.")
        return all_video_frames


def save_mp4_ffmpeg(frames, output_path, fps=16):
    """
    Pipes the PIL frames directly to ffmpeg to create an MP4.
    """
    if not frames:
        print("No frames to save.")
        return

    # Check if ffmpeg is installed
    if not shutil.which("ffmpeg"):
        print("❌ Error: 'ffmpeg' not found in PATH.")
        print("   Falling back to saving individual frames to directory.")

        # Fallback behavior
        base_dir = os.path.dirname(output_path)
        for i, image in enumerate(frames):
            filename = f"frame_{i:04d}.png"
            image.save(os.path.join(base_dir, filename))
        return

    print(f"🎬 Encoding {len(frames)} frames to {output_path} at {fps} FPS...")

    # ffmpeg command: read from pipe, output h264 mp4
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-f",
        "image2pipe",  # Input format
        "-vcodec",
        "png",  # Input codec
        "-r",
        str(fps),  # Frame rate
        "-i",
        "-",  # Input from stdin
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",  # Quality (lower is better)
        "-preset",
        "slow",
        output_path,
    ]

    try:
        # Open subprocess with piped stdin/stdout/stderr
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Write frames to ffmpeg stdin
        for frame in frames:
            with io.BytesIO() as buffer:
                frame.save(buffer, format="PNG")
                process.stdin.write(buffer.getvalue())

        # Close stdin and wait for finish
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"❌ FFmpeg Error:\n{stderr.decode()}")
        else:
            print(f"✅ Video saved successfully: {output_path}")

    except Exception as e:
        print(f"❌ Failed to process video: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.2 Video Generation Script")
    parser.add_argument("--proxy", type=str, required=True, help="RunPod proxy URL")
    parser.add_argument("--input", type=str, required=True, help="Local path to image")
    parser.add_argument("--prompt", type=str, help="Text prompt (optional if in JSON)")
    parser.add_argument("--lora-high", nargs="*", help="List of high noise LoRAs")
    parser.add_argument("--lora-low", nargs="*", help="List of low noise LoRAs")
    parser.add_argument("--length", type=int, default=81, help="Length for the video")
    parser.add_argument(
        "--segment-json", type=str, help="JSON string or path to JSON file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output", help="Directory to save output"
    )

    args = parser.parse_args()

    my_pil = PILImage.open(args.input)
    input_path = upload_to_comfy(my_pil, filename="runpod_input.png")

    video_frames = generate_video(
        proxy=args.proxy,
        input_path=input_path,
        segment_json=args.segment_json,
        prompt=args.prompt,
        lora_high=args.lora_high,
        lora_low=args.lora_low,
        length=args.length,
    )

    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = os.path.join(output_dir, f"{timestamp}.mp4")
        save_mp4_ffmpeg(
            video_frames, output_filename, fps=16
        )  # Wan2.2 standard is often 16fps

