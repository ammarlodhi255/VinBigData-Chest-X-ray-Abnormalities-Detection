import numpy as np
import pydicom
import os
from PIL import Image
import tqdm

input_dir = r'D:\Downloads\VinBig Chest Dataset\train'
output_dir = r'D:\Downloads\VinBig Chest Dataset\train2'

for file in tqdm.tqdm(os.listdir(input_dir)):
    filename = file[:file.find('.')]
    im = pydicom.dcmread(os.path.join(input_dir, file))
    im = im.pixel_array.astype(float)

    rescaled_image = (np.maximum(im, 0)/im.max()) * 255
    final_image = np.uint8(rescaled_image)

    final_image = Image.fromarray(final_image)
    # final_image.show()
    final_image.save(os.path.join(output_dir, filename + '.png'))
