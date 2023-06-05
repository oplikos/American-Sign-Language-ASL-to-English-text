import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import os
from GUI import GUI
from tkinter import *
from PIL import ImageTk, Image
from tensorflow.keras.optimizers import Adam
from spellchecker import SpellChecker


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
    def predict(classes, classifier, img):
        img = cv2.resize(img, (64, 64))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        pred = classifier.predict(img)
        return classes[np.argmax(pred)], pred


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

            print("Loading images from folder ", folder, " has started.")
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
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        ret, res = cv2.threshold(
            th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return res


def Auto_Correct(word):
    mySpellChecker = SpellChecker()
    return mySpellChecker.correction(word)


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = None

# pre-trained saved model with 99% accuracy
classifier = Model.load_classifier('classifier9872.h5')


def draw_region(image, center):
    cropped_image = cv2.rectangle(
        image,
        (center[0] - 130, center[1] - 130),
        (center[0] + 130, center[1] + 130),
        (0, 0, 255),
        2,
    )
    return cropped_image, image


def start_gui(title, size):
    gui = GUI(title, size)

    gui_frame = gui.create_frame(600, 600, 'ne', 1, 0, 'green')
    vid_label = Label(gui_frame)
    vid_label.grid()

    return gui, vid_label


def exit_app(gui, cap):
    gui.root.destroy()
    cap.release()


def update_frame(image, vid_label):
    image_fromarray = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image_fromarray)

    vid_label.imgtk = imgtk
    vid_label.config(image=imgtk)


def get_threshold(label_entrybox):
    value = label_entrybox.get('1.0', END)
    try:
        return float(value)
    except:
        return 0.95


def get_char(gesture):
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
               'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
               'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

    return Model.predict(classes, classifier, gesture)


def AddCharToWord(word, curr_char):
    temp_word = word
    if curr_char == 'space':
        # print(Auto_Correct(temp_word))
        temp_word = ""
    elif curr_char == 'del':
        temp_word = temp_word[0:-1]
        print('character has been deleted')
    elif curr_char != 'nothing':
        temp_word += curr_char.lower()
        print('character has been added: ', curr_char.lower())

    return [temp_word, curr_char]


def frame_video_stream(names, curr_char, prev_char, word, sentence, *args):
    kwargs = dict(zip(names, args))

    threshold = get_threshold(kwargs['th_box'])
    curr_char = curr_char
    prev_char = prev_char

    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    update_frame(image, kwargs['vid_label'])

    image.flags.writeable = False
    results = kwargs['hands'].process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            x = [landmark.x for landmark in hand_landmarks.landmark]
            y = [landmark.y for landmark in hand_landmarks.landmark]

            center = np.array(
                [np.mean(x) * image_width, np.mean(y) * image_height]).astype('int32')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cropped_img, full_img = draw_region(image, center)

            update_frame(full_img, kwargs['vid_label'])

            try:
                # print('from try')
                gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                gray = DataGatherer().edge_detection(gray)

                curr_char, pred = get_char(gray)
                char = cv2.putText(full_img, curr_char, (
                    center[0] - 135, center[1] - 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                char_prob = cv2.putText(full_img, '{0:.2f}'.format(np.max(
                    pred)), (center[0] + 60, center[1] - 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                update_frame(full_img, kwargs['vid_label'])

                kwargs['cc_box'].delete('1.0', 'end')
                kwargs['cc_box'].insert('end', curr_char)
                # print(curr_char)
                # compare the current char with the previous one and if matched, then won't add the current char
                # because the model catches the chars really quick and if the below if statement removed,
                # the current char will be added endlessly to the word

                # also we use the threshold to prevent the meaningless characters to be added to the word
                # as the program catches the motion of the user's hand when the user changes the gesture(the motion between the gestures)
                # and the program thinks
                # it's a gesture and tries to match it with some letter but with low probability
                if (curr_char != prev_char) and (np.max(pred) > threshold):
                    # the below print statement is related to the formatter
                    # print(pred)
                    temp = AddCharToWord(word, curr_char)
                    kwargs['ow_box'].insert('end', curr_char)

                    if (temp[0] == "") and (temp[1] != "del"):
                        sentence += Auto_Correct(word) + " "
                        kwargs['sent_box'].insert(
                            'end', Auto_Correct(word) + " ")
                        kwargs['ow_box'].delete('1.0', 'end')
                        kwargs['cw_box'].delete('1.0', 'end')
                        kwargs['cw_box'].insert('end', Auto_Correct(word))
                    word = temp[0]

                    prev_char = curr_char
            except:
                pass

    kwargs['vid_label'].after(
        1, frame_video_stream, names, curr_char, prev_char, word, sentence, *args)


def pipe_cam(gui, vid_label):

    curr_char = None
    prev_char = None
    word = ""
    sentence = ""

    # the predicted character won't be added to the word unless its
    # probability is higher than the threshold
    # in places with good brightness and a good camera, the threshold can be a high value
    # otherwise, it should be a low value and the reason for that is in places that meet
    # the above requirements, the model predicts the letters with a high probability to be
    # the correct letter the user meant to add
    threshold = float(0.95)

    # this formatter is to print the probability of the letters in a readable
    # format just for the programmers if they want to see what the probabilities are like
    float_formatter = "{:.5f}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})

    global cap
    cap = cv2.VideoCapture(0)

    labels_num = 5
    labels = ['threshold', 'current char',
              'original word', 'corrected word', 'sentence']

    Labels, entryboxes = gui.create_labels(
        labels_num, labels, 'nw', 0, 0, y_spacing=0.06, create_entrybox_per_label=1)

    entryboxes['original word_entrybox'].config(width=18)
    entryboxes['corrected word_entrybox'].config(width=18)
    entryboxes['sentence_entrybox'].config(width=18)

    entryboxes['threshold_entrybox'].insert(END, threshold)

    entryboxes['threshold_entrybox'].bind(
        "<Return>", lambda e: get_threshold(entryboxes['threshold_entrybox']))
    entryboxes['threshold_entrybox'].bind(
        "<KP_Enter>", lambda e: get_threshold(entryboxes['threshold_entrybox']))

    entryboxes['current char_entrybox'].config(state='disabled')

    entryboxes['original word_entrybox'].config(state='disabled')
    entryboxes['corrected word_entrybox'].config(state='disabled')
    entryboxes['sentence_entrybox'].config(state='disabled')

    kwargs = {
        'th_box': entryboxes['threshold_entrybox'],
        'cc_box': entryboxes['current char_entrybox'],
        'ow_box': entryboxes['original word_entrybox'],
        'cw_box': entryboxes['corrected word_entrybox'],
        'sent_box': entryboxes['sentence_entrybox'],
        'vid_label': vid_label,
        'hands': mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5),
    }

    frame_video_stream(labels, curr_char, prev_char, word, sentence, *kwargs)


gui, vid_label = start_gui("Gesture Recognition", "800x600")
pipe_cam(gui, vid_label)
gui.root.mainloop()

# release the video capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
