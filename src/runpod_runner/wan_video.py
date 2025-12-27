import json
import os
import random
import io
import requests
import subprocess
import shutil
from comfy_script.runtime import client, load, Workflow
import comfy_script.runtime.util as util


class WanVideoAutomation:
    def __init__(self, proxy_url):
        """
        Initializes the automation client and connects to the ComfyUI instance.
        """
        print(f"‼️ Connecting to ComfyUI at {proxy_url}...")
        load(proxy_url)

        import comfy_script.runtime.nodes as nodes

        self.nodes = nodes
        self.base_url = client.client.base_url

    def upload_to_comfy(self, pil_image, filename="script_input.png"):
        """
        Uploads a PIL image to the ComfyUI server's input directory via HTTP.
        Returns: The filename string expected by LoadImage.
        """
        byte_stream = io.BytesIO()
        pil_image.save(byte_stream, format="PNG")
        byte_stream.seek(0)

        base_url = self.base_url
        if not base_url.endswith("/"):
            base_url += "/"
        target_url = f"{base_url}upload/image"

        files = {"image": (filename, byte_stream, "image/png")}
        data = {"overwrite": "true"}

        response = requests.post(target_url, files=files, data=data)
        response.raise_for_status()

        return filename

    def _setup_models(self):
        clip = self.nodes.CLIPLoader("umt5_xxl_fp16.safetensors", "wan", "default")
        vae = self.nodes.VAELoader("wan_2.1_vae.safetensors")

        WAN_HIGH_NOISE_LIGHTNING_STRENGTH = 3.0
        WAN_LOW_NOISE_LIGHTNING_STRENGTH = 1.5

        wan_high_noise_model = self.nodes.UNETLoader(
            "Wan2.2/wan2.2_i2v_high_noise_14B_fp16.safetensors", "default"
        )
        wan_high_noise_model = self.nodes.LoraLoaderModelOnly(
            wan_high_noise_model,
            "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
            WAN_HIGH_NOISE_LIGHTNING_STRENGTH,
        )

        wan_low_noise_model = self.nodes.UNETLoader(
            "Wan2.2/wan2.2_i2v_low_noise_14B_fp16.safetensors", "default"
        )
        wan_low_noise_model = self.nodes.LoraLoaderModelOnly(
            wan_low_noise_model,
            "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
            WAN_LOW_NOISE_LIGHTNING_STRENGTH,
        )

        return clip, vae, wan_high_noise_model, wan_low_noise_model

    def _load_loras(self, model, loras):
        if isinstance(loras, str):
            loras = [loras]

        if not loras:
            return self.nodes.ModelSamplingSD3(model, 5)

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
        lora_model = self.nodes.LoraLoaderModelOnly(model, name, strength)

        if len(loras) >= 2:
            name, strength = parse_lora(loras[1])
            lora_model = self.nodes.LoraLoaderModelOnly(lora_model, name, strength)

        if len(loras) > 2:
            print("Note: Only first two LoRAs loaded.")

        lora_model = self.nodes.ModelSamplingSD3(lora_model, 5)
        return lora_model

    def _wan_frame_to_video(
        self,
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
        steps = self.nodes.PrimitiveInt(8)
        cfg_high = self.nodes.PrimitiveFloat(1)
        cfg_low = self.nodes.PrimitiveFloat(1)
        start_low_at_step = self.nodes.PrimitiveInt(4)

        if seed is None:
            seed = random.randint(0, 0xFFFFFFFFFFFFFF)

        empty_negative_conditioning = self.nodes.CLIPTextEncode("", clip)
        conditioning = self.nodes.CLIPTextEncode(prompt, clip)

        print(f"Generating segment with Prompt: '{prompt}'")
        print(f"High Noise LoRAs: {loras_high if loras_high else 'None (Base Model)'}")
        print(f"Seed: {seed}")

        lora_model_high = self._load_loras(model_high, loras_high)
        lora_model_low = self._load_loras(model_low, loras_low)

        width, height, _ = self.nodes.GetImageSize(start_image)

        if end_image is not None:
            positive, negative, latent = self.nodes.WanFirstLastFrameToVideo(
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
            positive, negative, latent = self.nodes.WanImageToVideo(
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

        latent = self.nodes.KSamplerAdvanced(
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
        latent = self.nodes.KSamplerAdvanced(
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

        segment1 = self.nodes.VAEDecode(latent, vae)
        return segment1

    def generate_video(
        self,
        input_path,
        segment_json=None,
        prompt=None,
        lora_high=None,
        lora_low=None,
        length=81,
    ):
        """
        Orchestrates the generation workflow.
        """

        with Workflow() as wf:
            clip, vae, wan_high_noise_model, wan_low_noise_model = self._setup_models()

            input_image, _ = self.nodes.LoadImage(input_path)

            config = {}

            if segment_json:
                try:
                    loaded_json = None

                    if os.path.isfile(segment_json):
                        with open(segment_json, "r") as f:
                            loaded_json = json.load(f)
                    else:

                        # This allows users to reference presets just by filename (e.g. "preset.json")
                        package_dir = os.path.dirname(os.path.abspath(__file__))
                        segments_path = os.path.join(
                            package_dir, "segments", segment_json
                        )

                        if os.path.isfile(segments_path):
                            print(f"‼️ Found segment JSON in package: {segments_path}")
                            with open(segments_path, "r") as f:
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

            start_image_node = input_image
            if "start_image" in config:
                print(
                    f"Loading explicit start image from JSON: {config['start_image']}"
                )
                start_image_node, _ = self.nodes.LoadImage(config["start_image"])

            end_image_node = None
            if "end_image" in config:
                print(f"Loading explicit end image: {config['end_image']}")
                end_image_node, _ = self.nodes.LoadImage(config["end_image"])

            if not final_prompt:
                print("Error: No prompt provided in JSON or CLI.")
                return []

            video_batch = self._wan_frame_to_video(
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

    def save_mp4_ffmpeg(self, frames, output_path, fps=16):
        """
        Pipes the PIL frames directly to ffmpeg to create an MP4.
        """
        if not frames:
            print("No frames to save.")
            return

        if not shutil.which("ffmpeg"):
            print("❌ Error: 'ffmpeg' not found in PATH.")
            print("   Falling back to saving individual frames to directory.")

            base_dir = os.path.dirname(output_path)
            for i, image in enumerate(frames):
                filename = f"frame_{i:04d}.png"
                image.save(os.path.join(base_dir, filename))
            return

        print(f"🎬 Encoding {len(frames)} frames to {output_path} at {fps} FPS...")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-r",
            str(fps),
            "-i",
            "-",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "slow",
            output_path,
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            for frame in frames:
                with io.BytesIO() as buffer:
                    frame.save(buffer, format="PNG")
                    process.stdin.write(buffer.getvalue())

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                print(f"❌ FFmpeg Error:\n{stderr.decode()}")
            else:
                print(f"✅ Video saved successfully: {output_path}")

        except Exception as e:
            print(f"❌ Failed to process video: {e}")