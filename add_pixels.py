import os

import numpy as np
from PIL import Image

image_path = 'datasets/vigna'

i = 0

for filename in os.listdir(image_path):
    image_array_new = []
    if filename[filename.rfind(".") + 1:] in ['jpg', 'jpeg', 'png']:
        image = Image.open(image_path + "/" + filename)
        image_array = np.array(image.getdata())
        one = []
        for t in range(28):
            one.append([image_array[t + 28 * 4][0], image_array[t + 28 * 4][1], image_array[t + 28 * 4][2]])
            one.append([image_array[t + 28 * 3][0], image_array[t + 28 * 3][1], image_array[t + 28 * 3][2]])
            one.append([image_array[t + 28 * 2][0], image_array[t + 28 * 2][1], image_array[t + 28 * 2][2]])
            one.append([image_array[t + 28][0], image_array[t + 28][1], image_array[t + 28][2]])
            one.append([image_array[t][0], image_array[t][1], image_array[t][2]])
            one.append([0, 0, 0])
            image_array_new.append(one)
            one = []
        pixels_array = np.asarray(image_array_new, dtype=np.uint8)
        new_image = Image.fromarray(pixels_array, mode="RGB").rotate(90, expand=True)
        new_image.save('datasets/vigna_new/' + filename)
    i += 1