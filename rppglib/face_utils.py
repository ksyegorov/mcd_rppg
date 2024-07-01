import mediapipe
import numpy as np
import cv2
from scipy.spatial import ConvexHull

NUM_LANDMARKS = 468

class MediaPipeFaceExtractor:
    def __init__(self, num_landmarks, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.num_landmarks = num_landmarks
        self.face_mesh = mediapipe.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
        self.result_dtype = np.int16
        
    def process_frame(self, frame):
        face_results = self.face_mesh.process(frame)
        assert face_results.multi_face_landmarks is not None, 'No face detected' 
        face_landmarks = face_results.multi_face_landmarks[0]
        frame_width, frame_height = frame.shape[0], frame.shape[1]
        points = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])
        points[:, 0] *= frame_height
        points[:, 1] *= frame_width
        return points.astype(self.result_dtype)

def detect_landmarks(video):
    lm_detector = MediaPipeFaceExtractor(NUM_LANDMARKS)
    points = list()
    for frame in video:
        lms = lm_detector.process_frame(frame)
        points.append(lms)
    points = np.array(points)
    return points

def find_bbox(landmarks):
    x_min = landmarks[:, :, 0].min()
    x_max = landmarks[:, :, 0].max()
    y_min = landmarks[:, :, 1].min()
    y_max = landmarks[:, :, 1].max()
    return x_min, x_max, y_min, y_max

def crop_video(video, bbox):
    x_min, x_max, y_min, y_max = bbox
    video = video[:, y_min: y_max, x_min: x_max]
    return video

def crop_landmarks(landmarks, bbox):
    x_min, x_max, y_min, y_max = bbox
    landmarks[:, :, 0] -= x_min
    landmarks[:, :, 1] -= y_min
    return landmarks

def get_convex_points(points):
    hull = ConvexHull(points)
    convex_points = points[hull.vertices].astype('int32')
    return convex_points

def get_mask(frame, poly_points):
    H, W = frame.shape[:2]
    mask = np.zeros((H, W), dtype='uint8')
    polys = [poly_points, ]
    cv2.fillPoly(mask, polys, 255)
    mask = mask > 125
    return mask

def process_video(video):
    landmarks = detect_landmarks(video)
    bbox = find_bbox(landmarks)
    video = crop_video(video, bbox)
    landmarks = crop_landmarks(landmarks, bbox)

    for i in range(video.shape[0]):
        points = landmarks[i]
        frame = video[i]
        convex = get_convex_points(points)
        mask = get_mask(frame, convex)
        video[i] *= mask[:, :, None]
    
    return video, landmarks