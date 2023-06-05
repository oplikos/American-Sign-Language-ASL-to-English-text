import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from spellchecker import SpellChecker
from newTe import Model


# Load the trained classifier
classifier_path = 'classifier.h5'
classifier = load_model(classifier_path)

# Define the classes corresponding to letters
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Initialize variables for tracking
sequence_length = 6  # Number of consecutive frames to consider
sequence = []

# Initialize spell checker
spell = SpellChecker()

# Function to predict the letter and perform spell checking


def predict_letter(frame):
    letter, _ = Model.predict(classes, classifier, frame)
    return letter

# Function to perform spell checking on the word


def spell_check(word):
    corrected_word = spell.correction(word)
    return corrected_word


# Open the video capture
cap = cv2.VideoCapture(0)

# Loop over frames
while True:
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret:
        break

    # Predict the letter
    letter = predict_letter(frame)

    # Append the letter to the sequence
    sequence.append(letter)

    # If the sequence has reached the desired length
    if len(sequence) == sequence_length:
        # Concatenate the letters in the sequence to form a word
        word = ''.join(sequence)

        # Perform spell checking on the word
        corrected_word = spell_check(word)

        # Print the predicted word
        print("Predicted Word:", corrected_word)

        # Reset the sequence
        sequence = []

    # Display the frame
    cv2.putText(frame, letter, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
