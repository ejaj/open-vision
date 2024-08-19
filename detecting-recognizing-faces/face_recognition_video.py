import os
import cv2
import numpy as np


def read_images_and_labels(path, total_images=311, images_per_person=None, sz=None):
    """
    Reads images from the specified path and returns arrays of images and labels.
    :param path: The directory path that contains the image files.
    :param total_images: Total number of images to read.
    :param images_per_person: Number of images per individual person.
    :param sz: A tuple specifying the size to resize images to (width, height).
    :return: A tuple (X, y) where X is a list of images and y is a list of labels.
    """
    X, y = [], []
    if images_per_person is None:
        images_per_person = total_images  # Default, assuming all images are of one person if not specified.

    current_label = 0
    image_count = 0

    for i in range(total_images):
        filepath = os.path.join(path, f"{i}.pgm")
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            if sz is not None:
                image = cv2.resize(image, sz)  # Resize the image if a size is specified.
            X.append(np.asarray(image, dtype=np.uint8))
            y.append(current_label)
            image_count += 1
            if image_count >= images_per_person:
                current_label += 1
                image_count = 0
        else:
            print(f"Failed to read image at {filepath}")

    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int32)


def main():
    path_to_training_images = 'data/training_faces'
    training_image_size = (200, 200)
    images_per_person = 1

    training_images, training_labels = read_images_and_labels(
        path_to_training_images, images_per_person=images_per_person, sz=training_image_size
    )

    names = [f"Person {i + 1}" for i in range(max(training_labels) + 1)]  # Create names dynamically based on the labels

    model = cv2.face.EigenFaceRecognizer_create()
    model.train(training_images, training_labels)

    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

    camera = cv2.VideoCapture(0)
    while (cv2.waitKey(1) == -1):
        success, frame = camera.read()
        if success:
            faces = face_cascade.detectMultiScale(frame, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_gray = gray[x:x + w, y:y + h]
                if roi_gray.size == 0:
                    continue
                roi_gray = cv2.resize(roi_gray, training_image_size)
                label, confidence = model.predict(roi_gray)
                if label < len(names):
                    text = f'{names[label]}, confidence={confidence:.2f}'
                    cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    print("Label index out of range, check the names array.")
            cv2.imshow('Face Recognition', frame)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
