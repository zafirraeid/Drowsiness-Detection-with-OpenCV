import cv2
import dlib
from scipy.spatial import distance
import winsound

def calculate_Ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

vid = cv2.VideoCapture(0)

face_detect = dlib.get_frontal_face_detector()
face_landmark = dlib.shape_predictor("Resources/shape_predictor_68_face_landmarks.dat")

while True:
    res, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect(gray)
    for face in faces:

        landmark = face_landmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = landmark.part(n).x
            y = landmark.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = landmark.part(next_point).x
            y2 = landmark.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = landmark.part(n).x
            y = landmark.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = landmark.part(next_point).x
            y2 = landmark.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = calculate_Ear(leftEye)
        right_ear = calculate_Ear(rightEye)

        EAR = (left_ear + right_ear) / 2
        EAR = round(EAR, 2)
        if EAR < 0.26:
            cv2.putText(frame, "DROWSY?", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
            cv2.putText(frame, "Wake Up!", (20, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            winsound.PlaySound("Resources/sound.mp3", winsound.SND_FILENAME)
            print("Drowsy")
        print(EAR)

    cv2.imshow("Drowsiness Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
vid.release()
cv2.destroyAllWindows()



