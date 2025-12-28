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
    parser.add_argument("--end-image", type=str, help="Path to end image (optional)")

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

    parser.add_argument(
        "--prepend",
        action="store_true",
        help="If set, generates video leading up to the input video. Requires --end-image to be set (acts as start image).",
    )


    parser.add_argument(
        "--loop",
        action="store_true",
        help="If set, creates a seamless loop. Prepend: Starts with LAST frame of input. Append: Ends with FIRST frame of input.",
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


    loop_start_pil = None
    loop_end_pil = None

    if is_video_input:
        print("📹 Detected MP4 input.")

        # If prepend is on and no time specified, assume start (0.0).
        # Otherwise default (None) implies end of video.
        extract_time = args.video_time
        if extract_time is None and args.prepend:
            extract_time = 0.0

        my_pil = automation.extract_frame(input_path, timestamp=extract_time)


        if args.prepend and args.loop:
            print(
                "🔄 Loop mode (Prepend): Extracting last frame of input video for generation start."
            )
            loop_start_pil = automation.extract_frame(input_path, timestamp=None)


        if not args.prepend and args.loop:
            print(
                "🔄 Loop mode (Append): Extracting first frame of input video for generation end."
            )
            loop_end_pil = automation.extract_frame(input_path, timestamp=0.0)
    else:
        my_pil = PILImage.open(input_path)

    upload_filename = os.path.basename(input_path)
    # If it was a video, we give the uploaded frame a specific name so we know it's a frame
    if is_video_input:
        upload_filename = f"{os.path.splitext(upload_filename)[0]}_frame.png"

    server_input_path = automation.upload_to_comfy(
        my_pil, filename=f"runpod_{upload_filename}"
    )


    server_loop_start_path = None
    if loop_start_pil:
        server_loop_start_path = automation.upload_to_comfy(
            loop_start_pil, filename=f"runpod_loop_start_{upload_filename}"
        )


    server_loop_end_path = None
    if loop_end_pil:
        server_loop_end_path = automation.upload_to_comfy(
            loop_end_pil, filename=f"runpod_loop_end_{upload_filename}"
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

    # Determine Generation Inputs based on Direction
    if args.prepend:
        # In prepend mode:
        # End Image = The frame extracted from the video (server_input_path)
        gen_end_path = server_input_path

        # Start Image Priority:
        # 1. Explicit --end-image (acts as start)
        # 2. --loop (last frame of video)
        # 3. Fallback (same as end image -> boomerang)
        if server_end_image_path:
            gen_start_path = server_end_image_path
        elif server_loop_start_path:
            gen_start_path = server_loop_start_path
        else:
            print(
                "⚠️ No start image or loop flag provided. Using input extracted frame as start (Start=End)."
            )
            gen_start_path = server_input_path

        print("🔄 Prepend Mode: Generating video LEADING UP TO input video frame.")
    else:
        # Standard Mode (Append):
        # Start Image = The frame extracted from video (or input image)
        gen_start_path = server_input_path


        # 1. Explicit --end-image
        # 2. --loop (first frame of video)
        # 3. None
        if server_end_image_path:
            gen_end_path = server_end_image_path
        elif server_loop_end_path:
            gen_end_path = server_loop_end_path
        else:
            gen_end_path = None

        if args.loop and not gen_end_path:
            print(
                "⚠️ Loop requested for Append but could not determine loop frame (is input a video?)."
            )

    video_frames = automation.generate_video(
        input_path=gen_start_path,
        segment=args.segment,
        prompt=args.prompt,
        lora_high=args.lora_high,
        lora_low=args.lora_low,
        length=args.length,
        seed=args.seed,
        end_image_path=gen_end_path,
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
            # Handle Concatenation Direction
            if args.prepend:
                # Order: Generated -> Input Video
                cut_point = args.video_time if args.video_time is not None else 0.0

                automation.concatenate_videos(
                    generated_filename,  # Video 1
                    input_path,  # Video 2
                    final_filename,
                    v1_cut=None,  # Don't cut generated
                    v2_start=cut_point,  # Start input video at cut point
                )
            else:
                # Order: Input Video -> Generated
                automation.concatenate_videos(
                    input_path,  # Video 1
                    generated_filename,  # Video 2
                    final_filename,
                    v1_cut=args.video_time,  # Cut input video at end
                    v2_start=None,  # Default logic (drop 1st frame)
                )


if __name__ == "__main__":
    main()