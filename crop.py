import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def center_crop(input_array):
    '''
    Only takes numpy arrays
    '''
    output_size = min(input_array.shape[:-1])
    start_row = (input_array.shape[0] - output_size) // 2 #1 is x, 0 is y
    end_row = input_array.shape[0] - start_row
    start_col = (input_array.shape[1] - output_size) // 2
    end_col = input_array.shape[1] - start_col
    output_array = input_array[start_row:end_row, start_col:end_col, ...]
    #print(input_array.shape)
    #print(output_array.shape)
    return output_array

# example
'''img1 = Image.open("test_imgs\IMG-2609.jpg")
img2 = Image.open("test_imgs/MicrosoftTeams-image.png")

plt.imshow(img1)
plt.show()
plt.imshow(center_crop(np.array(img1)))
plt.show()
plt.imshow(img2)
plt.show()
plt.imshow(center_crop(np.array(img2)))
plt.show()'''