import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


def load_images(path):
    images = []
    class_names = []
    ilist = os.listdir(path)
    for cl in ilist:
        cur_image = cv2.imread(f'{path}/{cl}')
        images.append(cur_image)
        class_names.append(os.path.splitext(cl)[0])
    return images, class_names


def find_encodings(images):
    encoded = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encoded.append(encode)
    return encoded


def mark_attendance(name):
    with open('att.txt', 'r+') as f:
        namelist = [line.split(',')[0] for line in f]
        if name not in namelist:
            now = datetime.now()
            dt_string = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{dt_string}')


path = 'image'
images, class_names = load_images(path)
encoded_list = find_encodings(images)
print('Encoded list completed')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    face_locs = face_recognition.face_locations(imgs)
    encodings = face_recognition.face_encodings(imgs, face_locs)

    for encoding, face_loc in zip(encodings, face_locs):
        matches = face_recognition.compare_faces(encoded_list, encoding)
        face_dis = face_recognition.face_distance(encoded_list, encoding)
        match_index = np.argmin(face_dis)

        if matches[match_index]:
            name = class_names[match_index].upper()
            y1, x2, y2, x1 = [coord * 4 for coord in face_loc]
            cv2.rectangle(imgs, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            mark_attendance(name)

    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
