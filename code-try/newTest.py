import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
model_path_Alpha = r"C:\Users\OPLIK\OneDrive\Desktop\SCHOOL\UCSD\Spring 2023\CSE 145\ASL-Translator\Model\Alpha\keras_model.h5"
model_label_Alpha = r"C:\Users\OPLIK\OneDrive\Desktop\SCHOOL\UCSD\Spring 2023\CSE 145\ASL-Translator\Model\Alpha\labels.txt"

classifier = Classifier(model_path_Alpha, model_label_Alpha)
offset = 20
imgSize = 300

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

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
            imgWhite = Image.fromarray(imgWhite)
            imgWhite = imgWhite.convert("RGB")
            imgWhite = np.array(imgWhite)
            imgWhite = imgWhite / 255.0
            imgWhite = np.expand_dims(imgWhite, axis=0)
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

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
            imgWhite = Image.fromarray(imgWhite)
            imgWhite = imgWhite.convert("RGB")
            imgWhite = np.array(imgWhite)
            imgWhite = imgWhite / 255.0
            imgWhite = np.expand_dims(imgWhite, axis=0)
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
        if index < len(labels):
            cv2.putText(imgOutput, labels[index], (x, y - 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            print("Prediction:", labels[index])
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)