import os
import cv2
import numpy as np


def normalize(X, low, high, dtype=None):
    """
    Normalizes a given array in X to a value between low and high.
    """
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    X = (X - minX) / (maxX - minX)  # normalize to [0...1]
    X = X * (high - low) + low  # scale to [low...high]
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


def main():
    image_path = 'data/training_faces'  # Ensure this path is correct and accessible
    [X, y] = read_images_and_labels(image_path)
    y = np.asarray(y, dtype=np.int32)

    model = cv2.face.EigenFaceRecognizer_create()
    model.train(np.asarray(X), np.asarray(y))

    [p_label, p_confidence] = model.predict(X[0])
    print(f"Predicted label = {p_label} (confidence={p_confidence:.2f})")

    # Getting mean and eigenvectors
    mean = model.getMean().reshape(X[0].shape)
    eigenvectors = model.getEigenVectors()

    mean_norm = normalize(mean, 0, 255, dtype=np.uint8)
    cv2.imshow("mean", mean_norm)
    cv2.waitKey(0)  # Display the mean image

    for i in range(min(len(X), 16)):  # Show up to 16 eigenfaces
        eigenvector_i = eigenvectors[:, i].reshape(X[0].shape)
        eigenvector_i_norm = normalize(eigenvector_i, 0, 255, dtype=np.uint8)
        cv2.imshow(f"eigenface_{i}", eigenvector_i_norm)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
