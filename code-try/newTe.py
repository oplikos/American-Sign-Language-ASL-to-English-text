from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

import cv2
import numpy as np


class Model:
    classifier = None

    def __init__(self, Type):
        self.classifier = Type

    @staticmethod
    def save_classifier(path, classifier):
        classifier.save(path)

    @staticmethod
    def load_classifier(path):
        classifier = load_model(path)
        return classifier

    @staticmethod
    def predict(classes, classifier, frame):
        # Preprocess the frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))  # Resize the image to the desired size
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        # Predict the class probabilities
        probabilities = classifier.predict(img)[0]
        predicted_class_index = np.argmax(probabilities)
        predicted_class = classes[predicted_class_index]

        return predicted_class, probabilities


class DataGatherer:
    def __init__(self, *args):
        if len(args) > 0:
            self.dir = args[0]
        elif len(args) == 0:
            self.dir = ""

    def load_images(self):
        images = []
        labels = []
        index = -1
        folders = sorted(os.listdir(self.dir))
        for folder in folders:
            index += 1
            print("Loading images from folder", folder, "has started.")
            for image in os.listdir(self.dir + '/' + folder):
                img = cv2.imread(self.dir + '/' + folder + '/' + image, 0)
                img = self.edge_detection(img)
                img = cv2.resize(img, (64, 64))
                img = img_to_array(img)
                images.append(img)
                labels.append(index)
        images = np.array(images)
        images = images.astype('float32') / 255.0
        labels = to_categorical(labels)
        x_train, x_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.1)
        return x_train, x_test, y_train, y_test

    def edge_detection(self, image):
        minValue = 70
        blur = cv2.GaussianBlur(image, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(
            th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return res


# training_dir = r'asl_alphabet_train'

# # loading the images from the training directory
# data_gatherer = DataGatherer(training_dir)
# x_train, x_test, y_train, y_test = data_gatherer.load_images()

# batch_size = 64
# training_size = x_train.shape[0]
# test_size = x_test.shape[0]

# # computing steps and validation steps per epoch according to training and testing size
# compute_steps_per_epoch = lambda x: int(np.ceil(1. * x / batch_size))
# steps_per_epoch = compute_steps_per_epoch(training_size)
# val_steps = compute_steps_per_epoch(test_size)

# # build the model
# classifier = Model.build_model(Sequential())

# classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # train the model
# history = classifier.fit(
#     x_train, y_train,
#     steps_per_epoch=steps_per_epoch,
#     epochs=35,
#     validation_data=(x_test, y_test),
#     validation_steps=val_steps
# )

# # save the classifier
# path = '/home/sbouzikian/private/Alpha/classifier.h5'
# Model.save_classifier(path, classifier)

# # plot accuracy graph
# plt.figure(figsize=(8, 5))
# plt.plot(history.history['accuracy'], label='train_accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend()
# plt.title("classifier")
# plt.show()
# plt.savefig('classifier.png')
