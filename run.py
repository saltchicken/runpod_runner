import argparse
import sys
from comfy_script.runtime import *
import datetime


parser = argparse.ArgumentParser(description="Wan2.2 Video Generation Script")

parser.add_argument("--proxy", type=str, required=True, help="RunPod proxy URL")
parser.add_argument("--input", type=str, required=True, help="Path to input image")
parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
parser.add_argument("--lora-high", nargs="*", required=True, help="List of high noise LoRAs")
parser.add_argument("--lora-low", nargs="*", required=True, help="List of low noise LoRAs")

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


def wan_first_last_frame_to_video(
    model_high,
    model_low,
    clip,
    vae,
    start_image,
    end_image,
    length,
    prompt,
    loras_high,
    loras_low,
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

    if not loras_high or not loras_low:
        #TODO: raise error
        return

    # NOTE: Debugging
    print(f"High Noise LoRAs: {loras_high}")

    if isinstance(loras_high, str):
        loras_high = [loras_high]

    if isinstance(loras_low, str):
        loras_low = [loras_low]

    # NOTE: This may be required because the base model (model_high) needs to remain unaltered.
    # LoraLoadModelOnly provides a deepcopy.
    lora_model_high = LoraLoaderModelOnly(model_high, loras_high[0], 1)

    if len(lora_high) == 2:
        lora_model_high = LoraLoaderModelOnly(lora_model_high, loras_high[1], 1)

    lora_model_high = ModelSamplingSD3(lora_model_high, 5)


    lora_model_low = LoraLoaderModelOnly(model_low, loras_low[0], 1)

    if len(loras_low) == 2:
        lora_model_low = LoraLoaderModelOnly(lora_model_low, loras_low[1], 1)

    lora_model_low = ModelSamplingSD3(lora_model_low, 5)

    width, height, _ = GetImageSize(start_image)

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

    latent = KSamplerAdvanced(
        lora_model_high,
        "enable",
        831186949035390,
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


    selected_frame, trimmed_batch = wan_first_last_frame_to_video(
        wan_high_noise_model,
        wan_low_noise_model,
        clip,
        vae,
        input_image,
        input_image,
        81,
        args.prompt,
        args.lora_high,
        args.lora_low,
    )

    merged = VideoMerge(trimmed_batch, None, None, None, None)

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
