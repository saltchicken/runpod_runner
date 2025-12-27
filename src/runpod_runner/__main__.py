import argparse
import os
import random
from datetime import datetime
from PIL import Image as PILImage
from dotenv import load_dotenv
from .wan_video import WanVideoAutomation


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Wan2.2 Video Generation Script")
    parser.add_argument("--proxy", type=str, required=True, help="RunPod proxy URL")

    parser.add_argument("--input", type=str, required=False, help="Local path to image")
    parser.add_argument("--prompt", type=str, help="Text prompt (optional if in JSON)")
    parser.add_argument("--lora-high", nargs="*", help="List of high noise LoRAs")
    parser.add_argument("--lora-low", nargs="*", help="List of low noise LoRAs")
    parser.add_argument("--length", type=int, default=81, help="Length for the video")
    parser.add_argument("--seed", type=int, default=None, help="Length for the video")

    parser.add_argument(
        "--segment",
        type=str,
        help="JSON string or path to JSON file",
    )

    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save output"
    )

    args = parser.parse_args()

    input_path = args.input
    env_input_dir = os.getenv("INPUT_DIR")

    if input_path is None:
        if env_input_dir and os.path.exists(env_input_dir):
            valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            files = [
                f
                for f in os.listdir(env_input_dir)
                if os.path.isfile(os.path.join(env_input_dir, f))
                and os.path.splitext(f)[1].lower() in valid_extensions
            ]
            if files:
                input_path = os.path.join(env_input_dir, random.choice(files))
                print(f"No input provided. Selected random file: {input_path}")
            else:
                raise FileNotFoundError(
                    f"No valid image files found in INPUT_DIR: {env_input_dir}"
                )
        else:
            raise ValueError(
                "Input not provided and INPUT_DIR environment variable is not set or valid."
            )

    if not os.path.exists(input_path) and env_input_dir:
        potential_path = os.path.join(env_input_dir, input_path)
        if os.path.exists(potential_path):
            input_path = potential_path
            print(f"Found input image in configured directory: {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found: {args.input or 'random selection'} (checked {input_path})"
        )

    automation = WanVideoAutomation(proxy_url=args.proxy)

    my_pil = PILImage.open(input_path)

    upload_filename = os.path.basename(input_path)
    server_input_path = automation.upload_to_comfy(
        my_pil, filename=f"runpod_{upload_filename}"
    )

    video_frames = automation.generate_video(
        input_path=server_input_path,
        segment=args.segment,
        prompt=args.prompt,
        lora_high=args.lora_high,
        lora_low=args.lora_low,
        length=args.length,
        seed=args.seed,
    )

    output_dir = args.output_dir or os.getenv("OUTPUT_DIR") or "./output"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = os.path.join(output_dir, f"{timestamp}.mp4")

        automation.save_mp4_ffmpeg(video_frames, output_filename, fps=16)


if __name__ == "__main__":
    main()
