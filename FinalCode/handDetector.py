from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from spellchecker import SpellChecker
import tensorflow as tf
import mediapipe as mp
from tkinter import *
import numpy as np
import cv2
import os
from PIL import ImageTk, Image

class GUI:

    def __init__(self, title, size):
        self.root = Tk()
        self.root.title(title)
        self.root.geometry(size)
        # self.root.attributes('-fullscreen', True)

    def create_frame(self, width, height, anchor, relx, rely, background='#37251b'):
        frame = Frame(self.root, bg=background, width=width, height=height)
        frame.place(anchor=anchor, relx=relx, rely=rely)
        return frame

    def create_labels(self, label_num, labels, anchor, relx, rely, x_spacing=0, y_spacing=0, create_entrybox_per_label=False):
        entry_labels = {}
        entry_boxes = {}
        relx = relx
        rely = rely
        longest_label_spacing = len(max(labels, key=len))/100.0
        for i in range(label_num):
            label = Label(self.root, text=labels[i]+": ",
                          font=("TimesNewRoman", 15))
            label.place(anchor=anchor, relx=relx, rely=rely)

            entry_labels[labels[i]] = label
            if create_entrybox_per_label:
                entry_box = Text(self.root, font=(
                    "TimesNewRoman", 20), height=1, width=10)
                entry_box.place(anchor=anchor, relx=relx +
                                longest_label_spacing+0.02, rely=rely)

                entry_boxes[labels[i]+'_entrybox'] = entry_box
            rely += y_spacing
            relx += x_spacing
        return entry_labels, entry_boxes

    def create_buttons(self, button_num, text, anchor, relx, rely, command=None, x_spacing=0, y_spacing=0):
        buttons = {}
        relx = relx
        rely = rely

        for i in range(button_num):
            btn = Button(self.root, command=command, text=text[i])
            btn.place(anchor=anchor, relx=relx, rely=rely)

            buttons[text[i]+' button'] = btn

            rely += y_spacing
            relx += x_spacing

        return buttons

    def load_images_from_dir(self, dir_path, image_size):
        images = []
        for file in os.listdir(dir_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(dir_path, file)
                image = Image.open(image_path)
                image = image.resize(image_size, Image.ANTIALIAS)
                image_tk = ImageTk.PhotoImage(image)
                images.append(image_tk)
        return images

class Model:

    classifier = None

    def __init__(self, Type):
        self.classifier = Type

    def build_model(classifier):
        classifier.add(Convolution2D(
            128, (3, 3), input_shape=(64, 64, 1), activation='relu'))
        classifier.add(Convolution2D(256, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Convolution2D(256, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Convolution2D(512, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Dropout(0.5))
        classifier.add(Convolution2D(512, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Dropout(0.5))
        classifier.add(Flatten())
        classifier.add(Dropout(0.5))
        classifier.add(Dense(1024, activation='relu'))
        classifier.add(Dense(29, activation='softmax'))

        return classifier

    def save_classifier(path, classifier):
        classifier.save(path)

    def load_classifier(path):
        classifier = load_model(path)
        return classifier

    def predict(classes, classifier, img):
        img = cv2.resize(img, (64, 64))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img/255.0

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
        images = images.astype('float32')/255.0
        labels = to_categorical(labels)

        x_train, x_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.1)

        return x_train, x_test, y_train, y_test

    def edge_detection(self, image):
        minValue = 80
        blur = cv2.GaussianBlur(image, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(
            th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return res

def Auto_Correct(word):
    mySpellChecker = SpellChecker()
    return mySpellChecker.correction(word)
    
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def draw_region(image, center):
    cropped_image = cv2.rectangle(image, (center[0] - 130, center[1] - 130),
                                  (center[0] + 130, center[1] + 130), (0, 150, 255), 5)
    return cropped_image[center[1]-130:center[1]+130, center[0]-130:center[0]+130], cropped_image

def start_gui(title, size):
    gui = GUI(title, size)
    gui_frame = gui.create_frame(600, 600, 'ne', 0.9, 0, '#37251b')
    vid_label = Label(gui_frame)
    vid_label.grid(row=0, column=0)
    # Load and resize images
    image_size = (300, 300)
    images = gui.load_images_from_dir("test", image_size)
    image_names = ['del', 'nothing', 'space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                   'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                   'W', 'X', 'Y', 'Z']
    image_label = Label(gui_frame, text="")
    image_label.grid(row=3, column=2, pady=5)
    # Function to update the image label
    def update_image_label(image_index):
        image_label.config(image=images[image_index])
        image_label.image = images[image_index]
    # Initial image index
    image_index = 0
    update_image_label(image_index)
    # Button to switch to the next image
    next_image_button = Button(
        gui_frame, text="Next", command=lambda: switch_image(1))
    next_image_button.grid(row=2, column=2, padx=5)
    # Button to switch to the previous image
    prev_image_button = Button(
        gui_frame, text="Previous", command=lambda: switch_image(-1))
    prev_image_button.grid(row=1, column=2, padx=5)

    # Function to switch between images
    def switch_image(step):
        nonlocal image_index
        image_index += step
        if image_index < 0:
            image_index = len(images) - 1
        elif image_index >= len(images):
            image_index = 0
        update_image_label(image_index)
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
        temp_word = " "
    elif curr_char == 'del':
        temp_word = temp_word[0:-1]
        print('character has been deleted')
    elif curr_char == 'nothing':
        print('')
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
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    update_frame(image, kwargs['vid_label'])
    image.flags.writeable = False
    results = kwargs['hands'].process(image)
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
                gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                gray = DataGatherer().edge_detection(gray)
                curr_char, pred = get_char(gray)
                char = cv2.putText(full_img, curr_char, (
                    center[0]-135, center[1]-135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                char_prob = cv2.putText(full_img, '{0:.2f}'.format(np.max(
                    pred)), (center[0]+60, center[1]-135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                update_frame(full_img, kwargs['vid_label'])
                kwargs['cc_box'].delete('1.0', 'end')
                kwargs['cc_box'].insert('end', curr_char)
                if (curr_char != prev_char) and (np.max(pred) > threshold):
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

def pipe_cam(gui, vid_label, images):
    curr_char = None
    prev_char = None
    word = ""
    sentence = ""
    threshold = float(0.95)
    float_formatter = "{:.5f}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})
    global cap
    cap = cv2.VideoCapture(0)
    labels_num = 5
    labels = ['threshold', 'current char',
              'original word', 'corrected word', 'sentence']
    Labels, entryboxes = gui.create_labels(
        labels_num, labels, 'nw', 00, 0.5, y_spacing=0.06, create_entrybox_per_label=True)
    entryboxes['threshold_entrybox'].config(width=25)
    entryboxes['current char_entrybox'].config(width=25)
    entryboxes['original word_entrybox'].config(width=25)
    entryboxes['corrected word_entrybox'].config(width=25)
    entryboxes['sentence_entrybox'].config(width=25)
    th_entrybox = entryboxes['threshold_entrybox']
    cc_entrybox = entryboxes['current char_entrybox']
    ow_entrybox = entryboxes['original word_entrybox']
    cw_entrybox = entryboxes['corrected word_entrybox']
    sent_entrybox = entryboxes['sentence_entrybox']
    names = ['vid_label', 'hands', 'th_box',
             'cc_box', 'ow_box', 'cw_box', 'sent_box']
    Exit_program_btn = gui.create_buttons(
        1, ['Exit'], 's', 0.5, 0.9, command=lambda: exit_app(gui, cap))
    with mp_hands.Hands(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.5,
            max_num_hands=1) as hands:
        frame_video_stream(names, curr_char, prev_char, word, sentence, vid_label,
                           hands,  th_entrybox, cc_entrybox, ow_entrybox, cw_entrybox, sent_entrybox)
        gui.root.configure(background="#37251b")
        gui.root.mainloop()
        

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = None
classifier = Model.load_classifier('grayscale_classifier1')
image_folder = 'test'
images = load_images_from_folder(image_folder)
title = "American-Sign-Language-ASL-to-English-text"
size = "1100x1100"
gui, vid_label = start_gui(title, size)
pipe_cam(gui, vid_label, images)
