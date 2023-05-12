import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('test_imgs\IMG-2609.jpg')
og_img = np.asarray(img)
# Convert to grayscale image with shape (32, 32, 1)
grayscale_image = np.dot(og_img, [0.2989, 0.5870, 0.1140])
grayscale_image = grayscale_image.astype(np.uint8).reshape((grayscale_image.shape[0],grayscale_image.shape[1],1))

grayscale_image = np.repeat(grayscale_image, 3, axis=2)

plt.imshow(og_img)
plt.show()
plt.imshow(grayscale_image)
plt.show()