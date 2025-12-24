import argparse
import json
import os
import random
from comfy_script.runtime import *
import datetime
import subprocess
import time

parser = argparse.ArgumentParser(description="Wan2.2 Video Generation Script")

parser.add_argument("--proxy", type=str, required=True, help="RunPod proxy URL")
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Local path to image, or remote path if already uploaded",
)

parser.add_argument(
    "--ssh-target", type=str, help="SSH user@host (e.g., root@123.456.78.9)"
)
parser.add_argument("--ssh-port", type=str, default="22", help="SSH port")
parser.add_argument("--ssh-key", type=str, help="Path to private SSH key")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./output",
    help="Local directory to save the video",
)

parser.add_argument("--prompt", type=str, help="Text prompt (optional if in JSON)")

parser.add_argument(
    "--lora-high",
    nargs="*",
    help="List of high noise LoRAs (format: name or name:strength)",
)
parser.add_argument(
    "--lora-low",
    nargs="*",
    help="List of low noise LoRAs (format: name or name:strength)",
)
parser.add_argument(
    "--length", type=int, default=81, help="Length for the first instance"
)
parser.add_argument(
    "--segments-json",
    type=str,
    help="JSON string or path to JSON file containing the list of segments.",
)

args = parser.parse_args()


# Determine if input is local or remote
remote_input_path = args.input

if os.path.exists(args.input):
    print(f"--- Detected local input file: {args.input} ---")
    if not args.ssh_target:
        print("Error: Local file found, but no --ssh-target provided to upload it.")
        exit(1)

    filename = os.path.basename(args.input)
    # The target directory on the remote server as defined in your prompt
    remote_dir = "/workspace/input"
    remote_input_path = f"{remote_dir}/{filename}"

    print(f"Uploading to {args.ssh_target}:{remote_input_path}...")

    scp_upload_cmd = ["scp", "-P", args.ssh_port]
    if args.ssh_key:
        scp_upload_cmd.extend(["-i", args.ssh_key])
        scp_upload_cmd.extend(["-o", "StrictHostKeyChecking=no"])

    # Upload source -> destination
    scp_upload_cmd.append(args.input)
    scp_upload_cmd.append(f"{args.ssh_target}:{remote_dir}/")

    try:
        subprocess.run(scp_upload_cmd, check=True)
        print("✅ Upload complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed with error code {e.returncode}")
        exit(1)
else:
    print(f"Input not found locally, assuming remote path: {args.input}")


load(args.proxy)
from comfy_script.runtime.nodes import *


def setup_models():
    clip = CLIPLoader("umt5_xxl_fp16.safetensors", "wan", "default")
    vae = VAELoader("wan_2.1_vae.safetensors")

    WAN_HIGH_NOISE_LIGHTNING_STRENGTH = 3.0
    WAN_LOW_NOISE_LIGHTNING_STRENGTH = 1.5

    wan_high_noise_model = UNETLoader(
        "Wan2.2/wan2.2_i2v_high_noise_14B_fp16.safetensors", "default"
    )
    wan_high_noise_model = LoraLoaderModelOnly(
        wan_high_noise_model,
        "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
        WAN_HIGH_NOISE_LIGHTNING_STRENGTH,
    )

    wan_low_noise_model = UNETLoader(
        "Wan2.2/wan2.2_i2v_low_noise_14B_fp16.safetensors", "default"
    )
    wan_low_noise_model = LoraLoaderModelOnly(
        wan_low_noise_model,
        "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
        WAN_LOW_NOISE_LIGHTNING_STRENGTH,
    )

    return clip, vae, wan_high_noise_model, wan_low_noise_model


def load_loras(model, loras):
    if isinstance(loras, str):
        loras = [loras]

    if not loras:
        return ModelSamplingSD3(model, 5)


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
    lora_model = LoraLoaderModelOnly(model, name, strength)


    if len(loras) >= 2:
        name, strength = parse_lora(loras[1])
        lora_model = LoraLoaderModelOnly(lora_model, name, strength)


    if len(loras) > 2:
        print("Note: Only first two LoRAs loaded.")

    lora_model = ModelSamplingSD3(lora_model, 5)
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
    nth_last_frame=1,
):
    steps = PrimitiveInt(8)
    cfg_high = PrimitiveFloat(1)
    cfg_low = PrimitiveFloat(1)
    start_low_at_step = PrimitiveInt(4)

    if seed is None:
        seed = random.randint(0, 0xFFFFFFFFFFFFFF)

    empty_negative_conditioning = CLIPTextEncode("", clip)
    conditioning = CLIPTextEncode(prompt, clip)

    print(f"Generating segment with Prompt: '{prompt}'")

    print(f"High Noise LoRAs: {loras_high if loras_high else 'None (Base Model)'}")
    print(f"Seed: {seed}")

    lora_model_high = load_loras(model_high, loras_high)
    lora_model_low = load_loras(model_low, loras_low)

    width, height, _ = GetImageSize(start_image)

    if end_image is not None:
        positive, negative, latent = WanFirstLastFrameToVideo(
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
        positive, negative, latent = WanImageToVideo(
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

    latent = KSamplerAdvanced(
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
    latent = KSamplerAdvanced(
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

    segment1 = VAEDecode(latent, vae)
    selected_frame, trimmed_batch = NthLastFrameSelector(segment1, nth_last_frame)
    return selected_frame, trimmed_batch


output = None

with Workflow() as wf:
    clip, vae, wan_high_noise_model, wan_low_noise_model = setup_models()

    # If uploaded, this now points to /workspace/input/filename.png
    input_image, _ = LoadImage(remote_input_path)

    segments_to_process = []

    if args.segments_json:
        try:
            if os.path.isfile(args.segments_json):
                with open(args.segments_json, "r") as f:
                    segments_to_process = json.load(f)
            else:
                segments_to_process = json.loads(args.segments_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            exit(1)
    elif args.prompt:
        segments_to_process = [
            {
                "prompt": args.prompt,
                "lora_high": args.lora_high,
                "lora_low": args.lora_low,
                "length": args.length,
            }
        ]
    else:
        print("Error: Provide --segments-json OR CLI args.")
        exit(1)

    generated_batches = []
    last_generated_frame = input_image
    current_lora_high = args.lora_high
    current_lora_low = args.lora_low

    for i, seg in enumerate(segments_to_process):
        print(f"\n--- Processing Segment {i + 1}/{len(segments_to_process)} ---")

        seg_prompt = seg.get("prompt")
        if not seg_prompt:
            print(f"Skipping segment {i + 1} (no prompt).")
            continue

        if "lora_high" in seg:
            current_lora_high = seg["lora_high"]
        if "lora_low" in seg:
            current_lora_low = seg["lora_low"]

        seg_length = seg.get("length", 81)
        seg_seed = seg.get("seed")
        seg_nth_last_frame = seg.get("nth_last_frame", 1)

        # Start Image Logic
        seg_start_image = last_generated_frame
        if "start_from_segment" in seg:
            target_idx = seg["start_from_segment"]
            target_nth = seg.get("start_nth_last", 1)
            if target_idx < len(generated_batches):
                print(f"Start Image: Segment {target_idx}, {target_nth}th last frame.")
                seg_start_image, _ = NthLastFrameSelector(
                    generated_batches[target_idx], target_nth
                )
            else:
                print(f"Error: start_from_segment {target_idx} invalid.")
                exit(1)
        elif "start_image" in seg:
            print(f"Loading explicit start image: {seg['start_image']}")
            seg_start_image, _ = LoadImage(seg["start_image"])

        # End Image Logic
        seg_end_image = None
        if "end_from_segment" in seg:
            target_idx = seg["end_from_segment"]
            target_nth = seg.get("end_nth_last", 1)
            if target_idx < len(generated_batches):
                print(f"End Image: Segment {target_idx}, {target_nth}th last frame.")
                seg_end_image, _ = NthLastFrameSelector(
                    generated_batches[target_idx], target_nth
                )
            else:
                print(f"Error: end_from_segment {target_idx} invalid.")
                exit(1)
        elif seg.get("use_last_frame_as_end", False):
            print("Using last generated frame as End Image.")
            seg_end_image = last_generated_frame
        elif "end_image" in seg:
            print(f"Loading explicit end image: {seg['end_image']}")
            seg_end_image, _ = LoadImage(seg["end_image"])

        # if current_lora_high is None or current_lora_low is None:
        #     print(f"Error: Segment {i + 1} missing loras.")
        #     exit(1)

        selected_frame, trimmed_batch = wan_frame_to_video(
            wan_high_noise_model,
            wan_low_noise_model,
            clip,
            vae,
            start_image=seg_start_image,
            prompt=seg_prompt,
            loras_high=current_lora_high,
            loras_low=current_lora_low,
            end_image=seg_end_image,
            length=seg_length,
            seed=seg_seed,
            nth_last_frame=seg_nth_last_frame,
        )

        generated_batches.append(trimmed_batch)
        last_generated_frame = selected_frame

    merge_inputs = generated_batches[:5]
    while len(merge_inputs) < 5:
        merge_inputs.append(None)

    if len(generated_batches) > 0:
        merged = VideoMerge(*merge_inputs)
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
        output_filename = f"wan22_16fps/scripted/{timestamp}"

        output = VHSVideoCombine(
            merged,
            16,
            0,
            output_filename,
            "video/h264-mp4",
            False,
            True,
            None,
            None,
            None,
        )
    else:
        print("No video generated.")

if output is not None:
    print("Waiting for VHSVideoCombine output...")

    # and fixes the None issue often seen with wf.task.wait() race conditions
    result = output.wait()

    output_path = None

    if isinstance(result, dict) and "gifs" in result:
        output_path = result["gifs"][0]["fullpath"]
    elif (
        hasattr(result, "_output")
        and result._output is not None
        and "gifs" in result._output
    ):
        output_path = result._output["gifs"][0]["fullpath"]
    elif isinstance(result, list) and len(result) > 0 and hasattr(result[0], "_output"):
        # Fallback for old style List[Result]
        output_path = result[0]._output["gifs"][0]["fullpath"]

    if output_path:
        if args.ssh_target:
            print(f"\n--- Retrieving video from {args.ssh_target} ---")
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            filename = os.path.basename(output_path)
            local_path = os.path.join(args.output_dir, filename)

            scp_cmd = ["scp", "-P", args.ssh_port]
            if args.ssh_key:
                scp_cmd.extend(["-i", args.ssh_key])
                scp_cmd.extend(["-o", "StrictHostKeyChecking=no"])

            scp_cmd.append(f"{args.ssh_target}:{output_path}")
            scp_cmd.append(local_path)

            print(f"Executing: {' '.join(scp_cmd)}")
            try:
                subprocess.run(scp_cmd, check=True)
                print(f"✅ Video successfully downloaded to: {local_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ SCP failed with error code {e.returncode}")
        else:
            print(f"Done. File is at (remote): {output_path}")
    else:
        print(f"❌ Could not extract output path. Result was: {result}")
else:
    print("No output node to wait on.")