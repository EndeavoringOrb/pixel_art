import matplotlib.pyplot as plt
from skimage import data
import numpy as np
from time import perf_counter
from skimage.transform import resize
import sentence_similarity as ss

a = ss.get_bert_sentence_similarity("a red car", "bird")
b = ss.get_bert_sentence_similarity("a red car", "metal")
print(a,b)

print("HI")

def add_noise(image, noise_level=0.075):
    """
    Returns a noisy image. 1 noise_level is 100% noise, 0 is 0% noise
    """
    noise = np.random.random(size=image.shape)
    probability = np.random.random(size=image.shape)
    mask = probability < noise_level
    new_image = np.copy(image)
    new_image[mask] = noise[mask]
    return new_image  # Clip values to [0, 1] range

# Load example image
image = data.astronaut()
#image = pixelation.pixelate(image,4)
image = resize(image,(32,32,3))

test_iters = 30

start = perf_counter()
# Add noise to the image
for i in range(test_iters):
    noisy_image = add_noise(image)
end = perf_counter()
print(f"Average Time to add noise: {(end-start)/test_iters}")

while True:
    # Plot the original and noisy images
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ax = axes.ravel()

    ax[0].imshow(image)
    ax[0].set_title('Original image')
    ax[1].imshow(noisy_image)
    ax[1].set_title('Noisy image')

    plt.tight_layout()
    plt.show()
    image = add_noise(image)
    noisy_image = add_noise(image)