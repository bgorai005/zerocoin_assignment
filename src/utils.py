from PIL import Image
import numpy as np

def load_image(path):
    image=Image.open(path).convert("RGBA")
    width, height =image.size
    img_np=np.array(image, dtype=np.uint8)
    return img_np, width, height

def save_image(data,path):
    image=Image.fromarray(data, mode="RGBA")
    image.save(path)
    print(f"Image saved tp {path}")

