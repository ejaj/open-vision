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
                    print("image " + filepath + " is none")
                    continue
                if sz is not None:
                    im = cv2.resize(im, sz)
                X.append(np.asarray(im, dtype=np.uint8))
                y.append(c)
            c += 1
    return X, y


if __name__ == "__main__":
    # Set a default path for images
    default_path = 'data/at'
    image_size = (200, 200)
    out_dir = 'data/stil'

    X, y = read_images(default_path, image_size)

    # Check if opencv-contrib-python is installed
    if hasattr(cv2, 'face'):
        model = cv2.face.EigenFaceRecognizer_create()
        model.train(np.asarray(X), np.asarray(y))

        p_label, p_confidence = model.predict(X[0])
        print("Predicted label = %d (confidence=%.2f)" % (p_label, p_confidence))

        mean = model.getMean().reshape(X[0].shape)
        cv2.imwrite(os.path.join(out_dir, "mean.png"), normalize(mean, 0, 255, dtype=np.uint8))

        eigenvectors = model.getEigenVectors()
        for i in range(min(10, len(X))):
            eigenface = eigenvectors[:, i].reshape(X[0].shape)
            cv2.imwrite(os.path.join(out_dir, f"eigenface_{i}.png"), normalize(eigenface, 0, 255, dtype=np.uint8))
    else:
        print("opencv-contrib-python is not installed. Please install it to use the face module.")
