import os
from PIL import Image
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import time
import tensorflow as tf

def process_images(data_folder, save_folder):
    data = []
    labels = []
    # Iterate over each image file in the folder
    for image_file in os.listdir(data_folder):
        image_path = os.path.join(data_folder, image_file)

        # Load the image
        image = Image.open(image_path)

        # Convert the image to grayscale
        # grayscale_image = image.convert('L')
        image_array = np.array(image)
        # Convert the grayscale image back to RGB mode
        # wireframe_image = grayscale_image.convert('RGB')
        data.append(image_array)
        # Save the wireframe image to the save folder
        save_path = os.path.join(save_folder, image_file)
        # wireframe_image.save(save_path)
    data = np.array(data, dtype='float32')
    labels = np.array(labels)
    data /= 255.0
    print("Wireframe images saved successfully!")


# Example usage:
data_folder = r'C:\Users\OPLIK\OneDrive\Desktop\asl_alphabet_train\del'
save_folder = r'C:\Users\OPLIK\OneDrive\Desktop\SCHOOL\UCSD\Spring 2023\CSE 145\ASL-Translator\Data\combienData\del'

process_images(data_folder, save_folder)
