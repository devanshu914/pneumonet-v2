import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

def preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img, dtype='float32')  # Ensure float32 from start
    img_array = np.expand_dims(img_array / 255.0, axis=0)  # Normalize inline
    return img_array
