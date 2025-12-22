import argparse
import json
import os
from comfy_script.runtime import *
import datetime


parser = argparse.ArgumentParser(description="Wan2.2 Video Generation Script")

parser.add_argument("--proxy", type=str, required=True, help="RunPod proxy URL")
parser.add_argument("--input", type=str, required=True, help="Path to input image")


parser.add_argument("--prompt", type=str, help="Text prompt (optional if in JSON)")
parser.add_argument(
    "--lora-high", nargs="*", help="List of high noise LoRAs (optional if in JSON)"
)
parser.add_argument(
    "--lora-low", nargs="*", help="List of low noise LoRAs (optional if in JSON)"
)
parser.add_argument(
    "--length", type=int, default=81, help="Length for the first instance"
)


parser.add_argument(
    "--segments-json",
    type=str,
    help="JSON string or path to JSON file containing the list of segments. "
    "Can replace CLI prompt/lora arguments completely.",
)

args = parser.parse_args()

load(args.proxy)
from comfy_script.runtime.nodes import *


def setup_models():
    clip = CLIPLoader("umt5_xxl_fp16.safetensors", "wan", "default")
    vae = VAELoader("wan_2.1_vae.safetensors")

    WAN_HIGH_NOISE_LIGHTNING_STRENGTH = 3.5

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

    # NOTE: This may be required because the base model (model_high) needs to remain unaltered.
    # LoraLoadModelOnly provides a deepcopy.
    if not loras:
        return ModelSamplingSD3(model, 5)

    lora_model = LoraLoaderModelOnly(model, loras[0], 1)

    if len(loras) == 2:
        lora_model = LoraLoaderModelOnly(lora_model, loras[1], 1)
    elif len(loras) > 2:
        print(
            "Please note that only the first two LoRAs will be loaded. Need to implement more lora loading logic."
        )

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
):
    steps = PrimitiveInt(8)
    cfg_high = PrimitiveFloat(1)
    cfg_low = PrimitiveFloat(1)
    start_low_at_step = PrimitiveInt(4)

    empty_negative_conditioning = CLIPTextEncode("", clip)
    conditioning = CLIPTextEncode(
        prompt,
        clip,
    )

    # NOTE: Debugging
    print(f"Generating segment with Prompt: '{prompt}'")
    print(f"High Noise LoRAs: {loras_high}")

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
        831186949035391,
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
    selected_frame, trimmed_batch = NthLastFrameSelector(segment1, 1)
    return selected_frame, trimmed_batch


with Workflow(wait=True):
    clip, vae, wan_high_noise_model, wan_low_noise_model = setup_models()

    input_image, _ = LoadImage(args.input)

    segments_to_process = []

    if args.segments_json:
        try:
            if os.path.isfile(args.segments_json):
                with open(args.segments_json, "r") as f:
                    segments_to_process = json.load(f)
            else:
                segments_to_process = json.loads(args.segments_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for segments: {e}")
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
        print(
            "Error: You must provide either --segments-json OR (--prompt, --lora-high, --lora-low)"
        )
        exit(1)

    generated_batches = []
    current_start_image = input_image

    current_lora_high = args.lora_high
    current_lora_low = args.lora_low

    for i, seg in enumerate(segments_to_process):
        print(f"\n--- Processing Segment {i + 1}/{len(segments_to_process)} ---")

        seg_prompt = seg.get("prompt")
        if not seg_prompt:
            print(f"Error: Segment {i + 1} is missing a 'prompt'. Skipping.")
            continue

        # If the segment has specific loras, use them AND update the current default.
        # If not, use the current default (which might be from previous segment).
        if "lora_high" in seg:
            current_lora_high = seg["lora_high"]
        if "lora_low" in seg:
            current_lora_low = seg["lora_low"]

        seg_length = seg.get("length", 81)

        # Validation: Ensure we have loras for the very first segment
        if current_lora_high is None or current_lora_low is None:
            print(
                f"Error: Segment {i + 1} needs lora-high/low defined (none inherited from CLI or previous)."
            )
            exit(1)

        selected_frame, trimmed_batch = wan_frame_to_video(
            wan_high_noise_model,
            wan_low_noise_model,
            clip,
            vae,
            start_image=current_start_image,
            prompt=seg_prompt,
            loras_high=current_lora_high,
            loras_low=current_lora_low,
            length=seg_length,
        )

        generated_batches.append(trimmed_batch)
        current_start_image = selected_frame

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
        print(output)
    else:
        print("No video generated.")

