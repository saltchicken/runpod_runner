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
    parser.add_argument("--seed", type=int, default=None, help="Seed for the video")
    parser.add_argument(
        "--end-image", type=str, help="Path to end image (optional)"
    )

    parser.add_argument(
        "--video-time",
        type=float,
        default=None,
        help="Timestamp (in seconds) to extract frame from and cut video at.",
    )

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

    automation = WanVideoAutomation(proxy_url=args.proxy)

    if input_path is None:
        if env_input_dir and os.path.exists(env_input_dir):
            valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".mp4"}
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

    is_video_input = input_path.lower().endswith(".mp4")

    if is_video_input:
        print("📹 Detected MP4 input.")

        my_pil = automation.extract_frame(input_path, timestamp=args.video_time)
    else:
        my_pil = PILImage.open(input_path)

    upload_filename = os.path.basename(input_path)
    # If it was a video, we give the uploaded frame a specific name so we know it's a frame
    if is_video_input:
        upload_filename = f"{os.path.splitext(upload_filename)[0]}_frame.png"

    server_input_path = automation.upload_to_comfy(
        my_pil, filename=f"runpod_{upload_filename}"
    )


    server_end_image_path = None
    if args.end_image:
        if not os.path.exists(args.end_image):
            raise FileNotFoundError(f"End image not found: {args.end_image}")

        print(f"📤 Uploading end image: {args.end_image}")
        end_pil = PILImage.open(args.end_image)
        end_filename = os.path.basename(args.end_image)
        server_end_image_path = automation.upload_to_comfy(
            end_pil, filename=f"runpod_end_{end_filename}"
        )

    video_frames = automation.generate_video(
        input_path=server_input_path,
        segment=args.segment,
        prompt=args.prompt,
        lora_high=args.lora_high,
        lora_low=args.lora_low,
        length=args.length,
        seed=args.seed,
        end_image_path=server_end_image_path,
    )

    output_dir = args.output_dir or os.getenv("OUTPUT_DIR") or "./output"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if is_video_input:
            # If splicing, the "generated" one is temporary/secondary
            generated_filename = os.path.join(output_dir, f"{timestamp}_generated.mp4")
            final_filename = os.path.join(output_dir, f"{timestamp}.mp4")
        else:
            final_filename = os.path.join(output_dir, f"{timestamp}.mp4")
            generated_filename = final_filename

        # Save the AI generated portion
        automation.save_mp4_ffmpeg(video_frames, generated_filename, fps=16)

        if is_video_input and os.path.exists(generated_filename):
            automation.concatenate_videos(
                input_path,
                generated_filename,
                final_filename,
                trim_duration=args.video_time,
            )


if __name__ == "__main__":
    main()