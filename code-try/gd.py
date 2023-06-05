import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from spellchecker import SpellChecker
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import pyttsx3
def text_to_speech(name):
    engine.say(name)
    engine.runAndWait()


engine = pyttsx3.init()
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

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

    def predict(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        probabilities = self.classifier.predict(img)[0]
        predicted_class_index = np.argmax(probabilities)
        predicted_class = classes[predicted_class_index]

        return predicted_class, probabilities


import cv2
import numpy as np
import math

offset = 20
imgSize = 300
counter = 0

# Load the trained classifier
classifier_path = 'classifier.h5'
classifier = load_model(classifier_path)

# Define the classes corresponding to letters
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Initialize variables for tracking
sequence_length = 6  # Number of consecutive frames to consider
sequence = []
model = Model(classifier)
# Initialize spell checker
spell = SpellChecker()
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)

            if imgCrop.size > 0:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            else:
                continue

            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            # prediction, index = classifier.getPrediction(imgWhite, draw=False)
            prediction, probabilities = model.predict(imgWhite)
            index = np.argmax(probabilities)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)

            if imgCrop.size > 0:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            else:
                continue

            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            # prediction, index = classifier.getPrediction(imgWhite, draw=False)
            prediction, probabilities = model.predict(imgWhite)
            index = np.argmax(probabilities)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
        if index < len(classes):
            # cv2.putText(imgOutput, labels[index], (x, y - 30),
            #             cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.putText(imgOutput, classes[index], (x, y - 30),
                cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            print("Prediction:", classes[index])

        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

