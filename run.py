from comfy_script.runtime import *

runpod_proxy = "https://355jbuu2d2i91q-8188.proxy.runpod.net/"
load("runpod_proxy")
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






def wan_first_last_frame_to_video(model_high, model_low, clip, vae, start_image, end_image, length, prompt, loras_high, loras_low):
    steps = PrimitiveInt(8)
    cfg_high = PrimitiveFloat(1)
    cfg_low = PrimitiveFloat(1)
    start_low_at_step = PrimitiveInt(4)

    empty_negative_conditioning = CLIPTextEncode("", clip)
    conditioning = CLIPTextEncode(prompt, clip,)

    # TODO: Add check if loras is a string or a list. If list then iterate, if string then load
    for i, lora in enumerate(loras_high):
        if i == 0:
            print("First pass")
            lora_model_high = LoraLoaderModelOnly(model_high, lora, 1)
        else:    
            print("Additional pass")
            lora_model_high = LoraLoaderModelOnly(lora_model_high, lora, 1)

    lora_model_high = ModelSamplingSD3(lora_model_high, 5)

    for i, lora in enumerate(loras_low):
        if i == 0:
            print("First pass")
            lora_model_low = LoraLoaderModelOnly(model_low, lora, 1)
        else:    
            print("Additional pass")
            lora_model_low = LoraLoaderModelOnly(lora_model_low, lora, 1)

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





with Workflow():

    clip, vae, wan_high_noise_model, wan_low_noise_model = setup_models()
    input_image, _ = LoadImage("/workspace/input/2025_12_22_094453_00009_.png")

    selected_frame, trimmed_batch = wan_first_last_frame_to_video(
        wan_high_noise_model,
        wan_low_noise_model,
        clip,
        vae,
        input_image,
        input_image,
        41,
        "The woman smiles",
        ["test_lora_high.safetensors"],
        ["test_lora_low.safetensors"],
    )


    merged = VideoMerge(trimmed_batch, None, None, None, None)
    output = VHSVideoCombine(
        merged,
        16,
        0,
        "wan22_16fps/scripted/2025_12_21_272353234234223423_2",
        "video/h264-mp4",
        False,
        True,
        None,
        None,
        None,
    )
    output.wait()
