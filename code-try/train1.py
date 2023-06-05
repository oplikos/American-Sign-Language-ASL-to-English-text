import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define input and output folders
input_folder = r"C:\Users\OPLIK\OneDrive\Desktop\SCHOOL\UCSD\Spring 2023\CSE 145\ASL-Translator\newData"
output_folder = r"C:\Users\OPLIK\OneDrive\Desktop\SCHOOL\UCSD\Spring 2023\CSE 145\ASL-Translator\trainData"

# Define model and training parameters
num_classes = 26  # Number of ASL alphabet classes
input_shape = (300, 300, 3)
batch_size = 32
epochs = 10

# Create an image data generator
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# Iterate through all folders in the input directory
for folder_name in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, folder_name)
    
    # Skip non-folder files
    if not os.path.isdir(folder_path):
        continue

    # Create train and validation data generators
    train_generator = datagen.flow_from_directory(
        folder_path,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        folder_path,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Create the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_generator,
              steps_per_epoch=train_generator.samples // batch_size,
              epochs=epochs,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples // batch_size)

    # Save the trained model
    output_model_path = os.path.join(output_folder, f"{folder_name}_model.h5")
    model.save(output_model_path)
    print(f"Saved model for {folder_name} alphabet: {output_model_path}")
