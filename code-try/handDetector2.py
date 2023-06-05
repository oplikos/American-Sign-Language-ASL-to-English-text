import mediapipe as mp
import cv2
import numpy as np
from keras.models import load_model
from Auto_Correct_SpellChecker import Auto_Correct
from GUI import GUI
from tkinter import *
from PIL import ImageTk, Image

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = None

# Pre-trained saved model with 99% accuracy
classifier = load_model('classifier9872.h5')


def draw_region(image, center):
    cropped_image = cv2.rectangle(image, (center[0] - 130, center[1] - 130),
                                  (center[0] + 130, center[1] + 130), (0, 0, 255), 2)
    return cropped_image[center[1]-130:center[1]+130, center[0]-130:center[0]+130], cropped_image


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

    prediction = classifier.predict(gesture[np.newaxis, ...])
    predicted_class = classes[np.argmax(prediction)]

    return predicted_class, prediction


def AddCharToWord(word, curr_char):
    temp_word = word
    if curr_char == 'space':
        temp_word = ""
    elif curr_char == 'del':
        temp_word = temp_word[0:-1]
        print('character has been deleted')
    elif curr_char != 'nothing':
        temp_word += curr_char.lower()
        print('character has been added:', curr_char.lower())

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
                gray = cv2.resize(gray, (64, 64))
                gray = gray.reshape((64, 64, 1))
                gray = gray / 255.0

                curr_char, pred = get_char(gray)
                char = cv2.putText(full_img, curr_char, (center[0]-135, center[1]-135),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                char_prob = cv2.putText(full_img, '{0:.2f}'.format(np.max(pred)),
                                        (center[0]+60, center[1]-135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

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


def pipe_cam(gui, vid_label):
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
        labels_num, labels, 'nw', 0, 0, y_spacing=0.06, create_entrybox_per_label=1)

    entryboxes['original word_entrybox'].config(width=18)
    entryboxes['corrected word_entrybox'].config(width=18)
    entryboxes['sentence_entrybox'].config(width=18, height=8)

    entryboxes['threshold_entrybox'].insert('end', threshold)
    th_entrybox = entryboxes['threshold_entrybox']

    cc_entrybox = entryboxes['current char_entrybox']
    ow_entrybox = entryboxes['original word_entrybox']
    cw_entrybox = entryboxes['corrected word_entrybox']
    sent_entrybox = entryboxes['sentence_entrybox']

    frame_video_stream(labels, curr_char, prev_char, word, sentence,
                       th_entrybox, cc_entrybox, ow_entrybox, cw_entrybox, sent_entrybox, vid_label)

    gui.root.protocol("WM_DELETE_WINDOW", lambda: exit_app(gui, cap))
    gui.root.mainloop()


def main():
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        gui, vid_label = start_gui("ASL Finger Spelling", "800x600")
        pipe_cam(gui, vid_label)


if __name__ == "__main__":
    main()
