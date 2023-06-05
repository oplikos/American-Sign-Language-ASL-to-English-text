import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
import time

# Define the folder path where the images are stored
data_folder = 'newData50'

# Define the input shape of the images
input_shape = (224, 224, 3)

# Define the number of classes (number of alphabets)
num_classes = 26

# Create an empty list to store the image data and labels
data = []
labels = []

# Assign a numeric label to each alphabet (0-25)
label_mapping = {chr(ord('A') + i): i for i in range(num_classes)}

# Iterate over each folder (letter) in the data folder
for folder in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, folder)

    # Iterate over each image file in the folder
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)

        # Load the image, resize it to the desired input shape, and convert to array
        image = Image.open(image_path).resize(input_shape[:2])
        image_array = np.array(image)

        # Add the image data and label to the lists
        data.append(image_array)
        labels.append(label_mapping[folder])
label_list = []
for label in labels:
    char = chr(label + ord('A'))
    if char not in label_list:
        label_list.append(char)
print(label_list)
# Convert the data and labels lists to NumPy arrays
data = np.array(data, dtype='float32')
labels = np.array(labels)

# Normalize the image data
data /= 255.0

# Convert the labels to one-hot encoded vectors
labels = utils.to_categorical(labels, num_classes)

# Shuffle the data and labels in the same order
indices = np.random.permutation(len(data))
data = data[indices]
labels = labels[indices]

# Split the data into training and testing sets
split_ratio = 0.6
split_index = int(len(data) * split_ratio)

x_train = data[:split_index]
y_train = labels[:split_index]
x_test = data[split_index:]
y_test = labels[split_index:]

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
start_time = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_test, y_test))
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# Save the trained model as a TensorFlow SavedModel
model.save('keras_model')

# Convert the model to TensorFlow Lite (TFLite) format

# Save the trained model
model.save('keras_model.h5')

# Save the labels to a file
with open('labels.txt', 'w') as file:
    file.write('\n'.join(label_list))