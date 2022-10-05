import cv2
import numpy as np
import dlib
from imutils.video import VideoStream, FPS

def index_nparray(nparray):
    index = None
    for num in nparray[0]:
            index = num
            break
    return index

def get_points_list(img, detector, predictor):
    graymask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #maska bolatyn suretti sur tuske auystyru
    faces  = detector(graymask)
    points_lists = []
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        landmarks = predictor(graymask, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points_lists += [(x, y)]
    return points_lists

def deepFake(img, img2, detector, predictor):
    points_list  = get_points_list(img, detector, predictor)
    #print(points_list)
    points_list2 = get_points_list(img2, detector, predictor)
    graymask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points = np.array(points_list, np.int32)
    #print(points)
    convexhull = cv2.convexHull(points)
    mask = np.zeros_like(graymask)
    cv2.fillConvexPoly(mask, convexhull, 255)
    face_image_1 = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points_list)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    triangles_id = []
    for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            id_pt1 = np.where((points == pt1).all(axis=1))
            id_pt1 = index_nparray(id_pt1)
            id_pt2 = np.where((points == pt2).all(axis=1))
            id_pt2 = index_nparray(id_pt2)
            id_pt3 = np.where((points == pt3).all(axis=1))
            id_pt3 = index_nparray(id_pt3)

            if id_pt1 is not None and id_pt2 is not None and id_pt3 is not None:
                triangle = [id_pt1, id_pt2, id_pt3]
                triangles_id.append(triangle)

    points2 = np.array(points_list2, np.int32)#
    #print(points2)
    convexhull2 = cv2.convexHull(points2)#
    img2_new_face = np.zeros_like(img2, np.uint8)
    for triangle_index in triangles_id:
            tr1_pt1 = points_list[triangle_index[0]]
            tr1_pt2 = points_list[triangle_index[1]]
            tr1_pt3 = points_list[triangle_index[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
            rect1 = cv2.boundingRect(triangle1)
            (x1, y1, w1, h1) = rect1
            cropped_triangle = img[y1: y1 + h1, x1: x1 + w1]
            cropped_tr1_mask = np.zeros((h1, w1), np.uint8)
            points = np.array([[tr1_pt1[0] - x1, tr1_pt1[1] - y1],
                      [tr1_pt2[0] - x1, tr1_pt2[1] - y1],
                      [tr1_pt3[0] - x1, tr1_pt3[1] - y1]], np.int32)
            cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
            cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_tr1_mask)

            tr2_pt1 = points_list2[triangle_index[0]]
            tr2_pt2 = points_list2[triangle_index[1]]
            tr2_pt3 = points_list2[triangle_index[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
            rect2 = cv2.boundingRect(triangle2)
            (x2, y2, w2, h2) = rect2
            cropped_triangle2 = img2[y2: y2 + h2, x2: x2 + w2]
            cropped_tr2_mask = np.zeros((h2, w2), np.uint8)
            points2 = np.array([[tr2_pt1[0] - x2, tr2_pt1[1] - y2],
                       [tr2_pt2[0] - x2, tr2_pt2[1] - y2],
                       [tr2_pt3[0] - x2, tr2_pt3[1] - y2]], np.int32)
            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
            cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2, mask=cropped_tr2_mask)

            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w2, h2),flags=cv2.INTER_NEAREST)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)


            img2_new_face_rect_area = img2_new_face[y2: y2 + h2, x2: x2 + w2]
            img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
    
            _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

            img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
            img2_new_face[y2: y2 + h2, x2: x2 + w2] = img2_new_face_rect_area

    img2_face_mask = np.zeros_like(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)
    img2_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = cv2.add(img2_noface, img2_new_face)

    (x3, y3, w3, h3) = cv2.boundingRect(convexhull2)
    center_face = (int((x3 + x3 + w3) / 2), int((y3 + y3 + h3) / 2))

    img2_face_mask = np.zeros_like(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)
    img2_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = cv2.add(img2_noface, img2_new_face)

    (x3, y3, w3, h3) = cv2.boundingRect(convexhull2)
    center_face = (int((x3 + x3 + w3) / 2), int((y3 + y3 + h3) / 2))
    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face, cv2.MONOCHROME_TRANSFER)
    return seamlessclone


def savevideo(imgpath, videopath):
    detector = dlib.get_frontal_face_detector() 
    predictor = dlib.shape_predictor('C:/Users/am1ri/Desktop/DeepFake/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat')
    img = cv2.imread(imgpath)
    cap = cv2.VideoCapture(videopath)
    if (cap.isOpened() == False): 
        print("Error reading video file")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    video_cod = cv2.VideoWriter_fourcc(*'XVID')
    #video_cod = cv2.VideoWriter_fourcc(*'mp4v')
    path = 'C:/Users/am1ri/Desktop/kivy project/output.avi'
    video_output = cv2.VideoWriter(path, video_cod, 30,(frame_width,frame_height))
    while True:
            ret,frame = cap.read()
            if ret == True:
                try:
                    deeps  = deepFake(img, frame, detector, predictor)
                    video_output.write(deeps)
                except:
                    pass
            else:
                break

    cap.release()
    return path





photo = 'C:\\Users\\am1ri\\Desktop\\DeepFake\\Media\\evans.jpg'
video = 'C:\\Users\\am1ri\\Desktop\\DeepFake\\Media\\Sups5.mp4'
print(savevideo(photo, video))
