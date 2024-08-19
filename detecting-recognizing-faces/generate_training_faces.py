import cv2
import os

output_folder = 'data/training_faces'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

face_classifier = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray, 1.3, 5, minSize=(120, 120))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        face_filename = '%s/%d.pgm' % (output_folder, count)
        cv2.imwrite(face_filename, face_img)
        count += 1
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break
cap.release()
cv2.destroyAllWindows()
