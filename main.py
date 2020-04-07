import sys
import numpy as np
from PIL import Image
import os

imagepath = "C:\\Users\\igorl\\Desktop\\csgoprints\\"
imagename = "img1.jpeg"

def read_img_array_RGB(imagepath, imagename):
    img = Image.open(imagepath + imagename).convert("RGB")
    img_array = np.array(img)
    return img_array


lista = os.listdir(imagepath)
lista = [e for e in lista if e[-4:]=="jpeg"]

image_tensor = {}

for i in lista:
    image_tensor[i] = read_img_array_RGB(imagepath, i)
    print(image_tensor[i].shape)
