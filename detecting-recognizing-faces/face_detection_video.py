import cv2

face_classifier = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray, 1.3, 5, minSize=(120, 120))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_classifier.detectMultiScale(
            roi_gray, 1.1, 5, minSize=(40, 40))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break
cap.release()
cv2.destroyAllWindows()
