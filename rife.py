import numpy as np
from PIL import Image
from rife_ncnn_vulkan_python import Rife

# Initialize RIFE (same as before)
rife = Rife(gpuid=0)
rife.load("rife-v4.9")

# input_images = [ ... your list of PIL objects ... ]
output_images = []

print(f"Processing {len(input_images)} frames with 4x interpolation...")

for i in range(len(input_images) - 1):
    img0 = input_images[i]
    img1 = input_images[i + 1]

    # Convert to Numpy
    np_img0 = np.array(img0)
    np_img1 = np.array(img1)

    # Add the original starting frame first
    output_images.append(img0)


    # 0.25 = 25% transition (Frame 1)
    # 0.50 = 50% transition (Frame 2)
    # 0.75 = 75% transition (Frame 3)
    for timestep in [0.25, 0.5, 0.75]:

        np_interp = rife.process(np_img0, np_img1, timestep=timestep)

        # Convert back to PIL and store
        img_interp = Image.fromarray(np_interp)
        output_images.append(img_interp)

# Append the very last original frame
output_images.append(input_images[-1])

print(f"Finished! New frame count: {len(output_images)}")

# Save (same as before)
import os

save_dir = "interpolated_frames_4x"
os.makedirs(save_dir, exist_ok=True)

for i, img in enumerate(output_images):
    img.save(os.path.join(save_dir, f"frame_{i:04d}.png"))