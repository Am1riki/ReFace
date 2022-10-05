import cv2
import dlib
import numpy as np

def facemarkdetectImage(path):
    frame = cv2.imread(path)
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

            landmarks = predictor(gframe, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        return frame

def Triangles(path):
    frame = cv2.imread(path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("C:/Users/am1ri/Desktop/kivy project/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")
    while True:
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gframe)
        mask = np.zeros_like(gframe)

        for face in faces:
            cv2.putText(frame, "{} face(s) found".format(len(faces)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            landmarks = predictor(gframe, face)
            marks_list = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                marks_list+=[(x,y)]
            
        points = np.array(marks_list, np.int32)
        contur = cv2.convexHull(points)

        rect = cv2.boundingRect(contur)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(marks_list)
        triangles = np.array(subdiv.getTriangleList(), dtype=np.int32)
        for t in triangles:
            pt1  = (t[0],t[1])
            pt2  = (t[2],t[3])
            pt3  = (t[4],t[5])

            cv2.line(frame, pt1, pt2, (0,0,255), 1)
            cv2.line(frame, pt2, pt3, (0,0,255), 1)
            cv2.line(frame, pt1, pt3, (0,0,255), 1)
        return frame

def getMask(path):
    frame = cv2.imread(path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("C:/Users/am1ri/Desktop/kivy project/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")
    while True:
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gframe)
        mask = np.zeros_like(gframe)

        for face in faces:
            cv2.putText(frame, "{} face(s) found".format(len(faces)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            landmarks = predictor(gframe, face)
            marks_list = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                marks_list+=[(x,y)]
        
        points = np.array(marks_list, np.int32)
        contur = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, contur, 255)
        face_image = cv2.bitwise_and(frame, frame, mask=mask)
        return face_image