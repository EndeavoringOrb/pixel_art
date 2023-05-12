from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")

import torch
import itertools
import math
import mediapy as media
import threading

image_reservoir = []
latents_reservoir = []


@torch.no_grad()
def plot_show_callback(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())
    image = pipeline.vae.decode(1 / 0.18215 * latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    # plt_show_image(image)
    plt.imsave(f"diffprocess/sample_{i:02d}.png", image)
    image_reservoir.append(image)


@torch.no_grad()
def save_latents(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())


################

prompt = "log cabin front view, 2.5D game"
negative_prompt = "ugly, malformed, modern"
num_steps = 50
num_images = 4

images = pipeline(prompt, 
                  num_inference_steps=num_steps, 
                  negative_prompt=negative_prompt, 
                  num_images_per_prompt=num_images,)

# Create a plot with num_images subplots
fig, axs = plt.subplots(1, num_images, figsize=(12, 4))

# Loop through the images and display them in subplots
for i, image in enumerate(images.images):
    axs[i].imshow(image)
    axs[i].axis("off")

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()