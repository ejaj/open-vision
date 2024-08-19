import sys

import cv2
import numpy as np
import os


def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    X = (X - minX) / (maxX - minX) * (high - low) + low
    return np.asarray(X, dtype=dtype) if dtype else np.asarray(X)


def read_images_and_labels(path, total_images=311, images_per_person=None):
    X, y = [], []
    if images_per_person is None:
        images_per_person = total_images

    current_label = 0
    image_count = 0

    for i in range(total_images):
        filepath = os.path.join(path, f"{i}.pgm")
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            X.append(np.asarray(image, dtype=np.uint8))
            y.append(current_label)
            image_count += 1
            if image_count >= images_per_person:
                current_label += 1
                image_count = 0
        else:
            print(f"Failed to read image at {filepath}")

    return X, y


def face_recognition_demo(image_path, output_dir=None):
    names = ['Kazi', 'Jane', 'Jack']

    # Load images and labels
    [X, y] = read_images_and_labels(image_path)
    y = np.asarray(y, dtype=np.int32)

    # Create an EigenFace Recognizer
    model = cv2.face.EigenFaceRecognizer_create()

    # Train the recognizer on the images and labels
    model.train(np.asarray(X), np.asarray(y))

    # Setup video capture
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

    while True:
        ret, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[x:x + w, y:y + h]
            try:
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi)
                print(f"Label: {names[params[0]]}, Confidence: {params[1]:.2f}")
                cv2.putText(img, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            except:
                continue

        cv2.imshow("Face Recognition", img)
        if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = 'data/training_faces'
    face_recognition_demo(image_path)
