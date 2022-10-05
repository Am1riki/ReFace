import cv2
import dlib

def facedetect(frame):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("C:/Users/am1ri/Desktop/kivy project/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")
    while True:
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gframe)
        for face in faces:
            cv2.putText(frame, "{} face(s) found".format(len(faces)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        return frame

