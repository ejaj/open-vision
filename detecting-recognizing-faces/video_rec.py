import os
import cv2
import numpy as np


def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    X = (X - minX) / (maxX - minX) * (high - low) + low
    return np.asarray(X, dtype=dtype) if dtype else np.asarray(X)


def read_images(path, sz=None):
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                filepath = os.path.join(subject_path, filename)
                im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if im is None:
                    print(f"Image {filepath} is None")
                    continue
                if sz is not None:
                    im = cv2.resize(im, sz)
                X.append(np.asarray(im, dtype=np.uint8))
                y.append(c)
            c += 1
    return X, y


def face_rec():
    names = ['Joe', 'Jane', 'Jack']
    path_to_images = 'path/to/your/image/dataset'  # Set your image dataset path here
    image_size = (200, 200)
    out_dir = 'data/stil'

    X, y = read_images(path_to_images, image_size)
    y = np.asarray(y, dtype=np.int32)

    model = cv2.face.EigenFaceRecognizer_create()
    model.train(np.asarray(X), np.asarray(y))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)

    while True:
        read, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[x:x + w, y:y + h]
            try:
                roi = cv2.resize(roi, image_size, interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi)
                print(f"Label: {params[0]}, Confidence: {params[1]:.2f}")
                cv2.putText(img, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except Exception as e:
                print(f"Error: {e}")
        cv2.imshow("camera", img)
        if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_rec()
