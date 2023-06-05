
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from PIL import Image
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
model_path_ABC = r"C:\Users\OPLIK\OneDrive\Desktop\SCHOOL\UCSD\Spring 2023\CSE 145\ASL-Translator\Model\ABC\keras_model.h5"
model_label_ABC = r"C:\Users\OPLIK\OneDrive\Desktop\SCHOOL\UCSD\Spring 2023\CSE 145\ASL-Translator\Model\ABC\labels.txt"
model_path_Aplphabet_front = r"C:\Users\OPLIK\OneDrive\Desktop\SCHOOL\UCSD\Spring 2023\CSE 145\ASL-Translator\Model\Alphabet_front\keras_model.h5"
model_label_Aplphabet_front = r"C:\Users\OPLIK\OneDrive\Desktop\SCHOOL\UCSD\Spring 2023\CSE 145\ASL-Translator\Model\Alphabet_front\labels.txt"
model_path_Aplphabet_back = r"C:\Users\OPLIK\OneDrive\Desktop\SCHOOL\UCSD\Spring 2023\CSE 145\ASL-Translator\Model\Alphabet_back\keras_model.h5"
model_label_Aplphabet_back = r"C:\Users\OPLIK\OneDrive\Desktop\SCHOOL\UCSD\Spring 2023\CSE 145\ASL-Translator\Model\Alphabet_back\labels.txt"
model_path_Alpha = r"C:\Users\OPLIK\OneDrive\Desktop\SCHOOL\UCSD\Spring 2023\CSE 145\ASL-Translator\Model\Alpha\keras_model.h5"
model_label_Alpha = r"C:\Users\OPLIK\OneDrive\Desktop\SCHOOL\UCSD\Spring 2023\CSE 145\ASL-Translator\Model\Alpha\labels.txt"

classifier = Classifier(model_path_Alpha, model_label_Alpha)
# classifier = Classifier(model_path_ABC, model_label_ABC)
# classifier = Classifier(model_path_Aplphabet_front, model_label_Aplphabet_front)
# classifier = Classifier(model_path_Aplphabet_back, model_label_Aplphabet_back)

offset = 20
imgSize = 300

folder = r"C:\Users\OPLIK\OneDrive\Desktop\SCHOOL\UCSD\Spring 2023\CSE 145\ASL-Translator\Data\C"
counter = 0

labels = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

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
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgWhite[:, wGap:wCal + wGap] = imgResize
            # imgWhite2 = cv2.resize(imgWhite, (64, 64))  # Resize imgWhite to (64, 64)
            # imgWhite2 = np.expand_dims(imgWhite2, axis=0)
            # prediction, index = classifier.getPrediction(imgWhite2, draw=False)
            imgWhite2 = cv2.resize(imgWhite, (64, 64))  # Resize imgWhite to (64, 64)
            if imgWhite2.size > 0:
                imgS = cv2.resize(imgWhite2, (224, 224))  # Resize imgWhite2 to (224, 224)
                imgS = np.expand_dims(imgS, axis=0)  # Add an extra dimension for batch size
                prediction, index = classifier.getPrediction(imgS, draw=False)
            k = imgSize / w
            hCal = math.ceil(k * h)
            if imgCrop.size > 0:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            else:
                continue
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgWhite[hGap:hCal + hGap, :] = imgResize
            # imgWhite2 = cv2.resize(imgWhite, (64, 64))  # Resize imgWhite to (64, 64)
            # imgWhite2 = np.expand_dims(imgWhite2, axis=0)
            # prediction, index = classifier.getPrediction(imgWhite2, draw=False)
            imgWhite2 = cv2.resize(imgWhite, (64, 64))  # Resize imgWhite to (64, 64)
            if imgWhite2.size > 0:
               imgS = cv2.resize(imgWhite2, (224, 224))  # Resize imgWhite2 to (224, 224)
               imgS = np.expand_dims(imgS, axis=0)  # Add an extra dimension for batch size
               prediction, index = classifier.getPrediction(imgS, draw=False)
                
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset), (255, 0, 255), cv2.FILLED)
        if index < len(labels):
            cv2.putText(imgOutput, labels[index], (x, y - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            print(" prediction " ,labels[index])
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
