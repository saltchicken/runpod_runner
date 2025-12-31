import json
import os
import random
import io
import requests
import subprocess
import shutil
from PIL import Image
import time
from comfy_script.runtime import client, load, Workflow
import comfy_script.runtime.util as util
from .utils import quiet


class WanVideoAutomation:
    def __init__(self, proxy_url):
        """
        Initializes the automation client and connects to the ComfyUI instance.
        """
        print(f"Connecting to ComfyUI at {proxy_url}...")
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

    def extract_frame(self, video_path, timestamp=None):
        """
        Extracts a frame from an MP4 video.
        If timestamp is None, extracts the last frame.
        If timestamp is provided (float/int), extracts the frame at that second.
        Returns PIL Image.
        """
        target_desc = f"{timestamp}s" if timestamp is not None else "end"
        print(f"🎞️ Extracting frame from: {video_path} at {target_desc}")

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_filename = tmp_file.name

        cmd = ["ffmpeg", "-y"]

        if timestamp is not None:
            # -ss before -i is fast seeking.
            cmd.extend(["-ss", str(timestamp), "-i", video_path, "-frames:v", "1"])
        else:
            # Old logic for last frame
            cmd.extend(["-sseof", "-1", "-i", video_path, "-update", "1"])

        cmd.extend(["-q:v", "2", tmp_filename])

        try:
            with quiet():
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            with open(tmp_filename, "rb") as f:
                img_data = f.read()

            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            return img

        except subprocess.CalledProcessError as e:
            print(f"❌ Error extracting frame: {e}")
            raise RuntimeError(f"Failed to extract frame from video at {target_desc}.")
        finally:
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)

    def concatenate_videos(
        self, video1_path, video2_path, output_path, v1_cut=None, v2_start=None
    ):
        """
        Concatenates two video files.
        v1_cut: If set, Video 1 is trimmed to this duration (from start).
        v2_start: If set, Video 2 starts from this timestamp.
        """
        print(
            f"🔗 Splicing videos:\n  1. {video1_path} (Cut At: {v1_cut})\n  2. {video2_path} (Start At: {v2_start})\n  -> {output_path}"
        )

        cmd = ["ffmpeg", "-y", "-i", video1_path, "-i", video2_path]

        # Video 1 Filter
        v1_filter = "[0:v]setpts=PTS-STARTPTS[v0];"
        if v1_cut is not None:
            v1_filter = f"[0:v]trim=duration={v1_cut},setpts=PTS-STARTPTS[v0];"

        # Video 2 Filter
        # We always attempt to drop 1 frame of overlap for smoothness if just appending.
        # If v2_start is provided, we seek to that point first.

        v2_trim_cmd = ""
        if v2_start is not None:
            # Trim from start point, then drop 1 frame to avoid duplication of the connection frame
            # chaining trims: first trim gets the segment, second trim removes 1st frame of that segment
            v2_trim_cmd = (
                f"trim=start={v2_start},setpts=PTS-STARTPTS,trim=start_frame=1"
            )
        else:
            # Standard: just drop first frame
            v2_trim_cmd = "trim=start_frame=1"

        v2_filter = f"[1:v]{v2_trim_cmd},setpts=PTS-STARTPTS[v1];"

        filter_str = f"{v1_filter}{v2_filter}[v0][v1]concat=n=2:v=1:a=0[outv]"

        cmd.extend(
            [
                "-filter_complex",
                filter_str,
                "-map",
                "[outv]",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "slow",
                "-crf",
                "18",
                output_path,
            ]
        )

        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            print(f"✅ Splicing complete: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error splicing videos: {e.stderr.decode()}")

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


        current_model = model

        if not loras:
            print("  - No additional LoRAs to load.")
            return self.nodes.ModelSamplingSD3(current_model, 5)

        def parse_lora(lora_str):
            lora_str = str(lora_str).strip()
            name = lora_str
            strength = 1.0
            if ":" in lora_str:
                parts = lora_str.rsplit(":", 1)
                try:
                    strength = float(parts[1])
                    name = parts[0].strip()
                except ValueError:
                    pass
            return name, strength

        print(f"  - Loading {len(loras)} LoRA(s)...")


        for i, lora_entry in enumerate(loras):
            name, strength = parse_lora(lora_entry)

            if not name:
                continue

            print(
                f"    [{i + 1}/{len(loras)}] + Loading LoRA: {name} (Strength: {strength})"
            )

            current_model = self.nodes.LoraLoaderModelOnly(
                current_model, name, strength
            )

        current_model = self.nodes.ModelSamplingSD3(current_model, 5)
        return current_model

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
            cfg_high,
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
        segment=None,
        prompt=None,
        lora_high=None,
        lora_low=None,
        length=81,
        seed=None,
        end_image_path=None,
    ):
        """
        Orchestrates the generation workflow.
        """

        with Workflow() as wf:
            clip, vae, wan_high_noise_model, wan_low_noise_model = self._setup_models()

            input_image, _ = self.nodes.LoadImage(input_path)

            config = {}

            if segment:
                try:
                    loaded_json = None

                    # 1. Check if exact file path exists
                    if os.path.isfile(segment):
                        with open(segment, "r") as f:
                            loaded_json = json.load(f)

                    elif os.path.isfile(f"{segment}.json"):
                        with open(f"{segment}.json", "r") as f:
                            loaded_json = json.load(f)
                    else:
                        # This allows users to reference presets just by filename (e.g. "preset" or "preset.json")
                        package_dir = os.path.dirname(os.path.abspath(__file__))

                        segments_path = os.path.join(package_dir, "segments", segment)
                        segments_path_ext = os.path.join(
                            package_dir, "segments", f"{segment}.json"
                        )

                        # 3. Check segments folder (exact)
                        if os.path.isfile(segments_path):
                            print(f"Found segment JSON in package: {segments_path}")
                            with open(segments_path, "r") as f:
                                loaded_json = json.load(f)

                        elif os.path.isfile(segments_path_ext):
                            print(f"Found segment JSON in package: {segments_path_ext}")
                            with open(segments_path_ext, "r") as f:
                                loaded_json = json.load(f)
                        else:
                            # 5. Fallback: Parse as raw JSON string
                            loaded_json = json.loads(segment)

                    if isinstance(loaded_json, list):
                        if len(loaded_json) > 0:
                            config = loaded_json[0]
                    elif isinstance(loaded_json, dict):
                        config = loaded_json
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    exit(1)


            final_prompt = prompt if prompt is not None else config.get("prompt")

            final_lora_high = config.get("lora_high", lora_high)
            final_lora_low = config.get("lora_low", lora_low)
            final_length = length
            final_seed = seed

            start_image_node = input_image
            if "start_image" in config:
                print(
                    f"Loading explicit start image from JSON: {config['start_image']}"
                )
                start_image_node, _ = self.nodes.LoadImage(config["start_image"])

            end_image_node = None

            target_end_image_path = end_image_path
            if target_end_image_path is None and "end_image" in config:
                target_end_image_path = config["end_image"]

            if target_end_image_path:
                print(f"Loading end image: {target_end_image_path}")
                end_image_node, _ = self.nodes.LoadImage(target_end_image_path)

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
            all_video_frames = []
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    all_video_frames = util.get_images(video_batch)
                    # Simple check to ensure we actually got data
                    if all_video_frames:
                        break
                except Exception as e:
                    print(
                        f"⚠️ Retrieval failed (Attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    # Catch the specific 'unexpected mimetype' error or 404s
                    if attempt < max_retries - 1:
                        sleep_time = 3 * (attempt + 1)
                        print(f"⏳ Waiting {sleep_time}s for proxy to sync...")
                        time.sleep(sleep_time)
                    else:
                        print("❌ Failed to retrieve images after multiple attempts.")
                        raise e

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
            print("    Falling back to saving individual frames to directory.")

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
            "yuv444p",
            "-crf",
            "0",
            "-preset",
            "veryslow",
            output_path,
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if process.stdin is None:
                raise RuntimeError("Failed to open stdin for ffmpeg process")

            with quiet():
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