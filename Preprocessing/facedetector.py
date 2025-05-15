# preprocessing/face_detector.py
from mtcnn import MTCNN
import cv2

detector = MTCNN()

def detect_faces_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detector.detect_faces(frame)
        if faces:
            x, y, w, h = faces[0]['box']
            face = frame[y:y+h, x:x+w]
            frames.append(cv2.resize(face, (224, 224)))
    cap.release()
    return frames
