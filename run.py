import argparse
from comfy_script.runtime import *
import datetime


parser = argparse.ArgumentParser(description="Wan2.2 Video Generation Script")

parser.add_argument("--proxy", type=str, required=True, help="RunPod proxy URL")
parser.add_argument("--input", type=str, required=True, help="Path to input image")
parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
parser.add_argument(
    "--lora-high", nargs="*", required=True, help="List of high noise LoRAs"
)
parser.add_argument(
    "--lora-low", nargs="*", required=True, help="List of low noise LoRAs"
)

parser.add_argument("--prompt2", type=str, help="Text prompt for the second instance")
parser.add_argument(
    "--lora-high2", nargs="*", help="List of high noise LoRAs for the second instance"
)
parser.add_argument(
    "--lora-low2", nargs="*", help="List of low noise LoRAs for the second instance"
)
parser.add_argument(
    "--length2", type=int, default=81, help="Length for the second instance"
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

    if not loras_high or not loras_low:
        # TODO: raise error
        return

    # NOTE: Debugging
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

    selected_frame, trimmed_batch = wan_frame_to_video(
        wan_high_noise_model,
        wan_low_noise_model,
        clip,
        vae,
        input_image,
        args.prompt,
        args.lora_high,
        args.lora_low,
        # end_image=input_image,
        length=81,
    )

    if args.prompt2:
        print("Generating second video segment...")

        # Use provided parameters or fallback to the first instance's parameters
        next_lora_high = args.lora_high2 if args.lora_high2 else args.lora_high
        next_lora_low = args.lora_low2 if args.lora_low2 else args.lora_low

        # Feed selected_frame (last frame of first batch) as start_image for next batch
        selected_frame_2, trimmed_batch_2 = wan_frame_to_video(
            wan_high_noise_model,
            wan_low_noise_model,
            clip,
            vae,
            start_image=selected_frame,
            prompt=args.prompt2,
            loras_high=next_lora_high,
            loras_low=next_lora_low,
            length=args.length2,
        )

        merged = VideoMerge(trimmed_batch, trimmed_batch_2, None, None, None)
    else:
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
