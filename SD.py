from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")

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