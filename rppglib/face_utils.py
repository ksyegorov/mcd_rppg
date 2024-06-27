import mediapipe
import cv2
import numpy as np
from scipy.spatial import ConvexHull

FACE_SEG = np.array([338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10])
FOREHEAD = np.array([10, 338, 297, 332, 333, 334, 296, 336, 9, 107, 66, 105, 63, 68, 54, 103, 67, 109])
LOW_FOREHEAD = np.array([151, 337, 299, 333, 334, 296, 336, 9, 107, 66, 105, 104, 69, 108])
NOSE = np.array([168, 417, 351, 419, 248, 281, 363, 344, 438, 457, 274, 354, 19, 125, 44, 237, 218, 115, 134, 51, 3, 196, 122, 193])
UPPER_LEFT_CHEEK  = np.array([116, 117, 118, 119, 120, 100, 142, 203, 206, 207, 147, 123])
UPPER_RIGHT_CHEEK = np.array([345, 346, 347, 348, 349, 329, 371, 423, 426, 427, 376, 352])

FACE_PARTS = dict()
FACE_PARTS['FACE_SEG'] = FACE_SEG
FACE_PARTS['LOW_FOREHEAD'] = LOW_FOREHEAD
FACE_PARTS['NOSE'] = NOSE
FACE_PARTS['UPPER_LEFT_CHEEK'] = UPPER_LEFT_CHEEK
FACE_PARTS['UPPER_RIGHT_CHEEK'] = UPPER_RIGHT_CHEEK
FACE_PARTS['FOREHEAD'] = FOREHEAD


class MediaPipeFaceExtractor:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.face_mesh = mediapipe.solutions.face_mesh.FaceMesh(max_num_faces=self.max_num_faces, 
                                                                min_detection_confidence=self.min_detection_confidence, 
                                                                min_tracking_confidence=self.min_tracking_confidence)
        
    def process_frame(self, frame):
        face_results = self.face_mesh.process(frame)
        if face_results.multi_face_landmarks is None:
            return None, None
        points, bbox = points_and_bbox(face_results, frame.shape[0], frame.shape[1])
        return points, bbox


def points_and_bbox(face_results, frame_height, frame_width):
    points = list()
    for face_landmarks in face_results.multi_face_landmarks:
        x1 = y1 = 1
        x2 = y2 = 0
        for lm in face_landmarks.landmark:
            cx, cy = lm.x, lm.y
            ccx = int(cx * frame_width)
            ccy = int(cy * frame_height)
            points.append((ccx, ccy))
            if cx < x1:
                x1 = cx
            if cy < y1:
                y1 = cy
            if cx > x2:
                x2 = cx
            if cy > y2:
                y2 = cy
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        x1, x2 = int(x1 * frame_width), int(x2 * frame_width)
        y1, y2 = int(y1 * frame_height), int(y2 * frame_height)
    return points, (x1, x2, y1, y2)


def plot_facebox(frame, bbox, color=(0, 0, 255), thickness=2):
    x1, x2, y1, y2 = bbox            
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def plot_facepoints(frame, points, thickness=1, clr=(0, 0, 255)):           
    for point in points:
        cv2.circle(frame, point, 1, clr, thickness)


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


def crop_face_video(video):
    face_detector = MediaPipeFaceExtractor()
    bboxes = list()
    for frame in video:
        points, bbox = face_detector.process_frame(frame)
        points = np.array(points).astype('int32')
        convex_points = get_convex_points(points)
        mask = get_mask(frame, convex_points)
        frame *= mask[:, :, None]
        bboxes.append(bbox)
    x1, x2, y1, y2 = pick_bbox(bboxes, expand=1.0)
    cropped_video = video[:, y1:y2, x1: x2, :].copy()
    return cropped_video


def pick_bbox(video_bboxes, expand=1.2):
    min_y1 = min([y1 for x1, x2, y1, y2 in video_bboxes])
    max_y2 = max([y2 for x1, x2, y1, y2 in video_bboxes])

    min_x1 = min([x1 for x1, x2, y1, y2 in video_bboxes])
    max_x2 = max([x2 for x1, x2, y1, y2 in video_bboxes])

    W = max_x2 - min_x1
    H = max_y2 - min_y1
    
    crop_size = max(W, H) * expand

    cen_x = min_x1 + W//2
    cen_y = min_y1 + H//2

    new_x1 = int(cen_x - crop_size/2)
    new_x2 = int(cen_x + crop_size/2)

    new_y1 = int(cen_y - crop_size/2)
    new_y2 = int(cen_y + crop_size/2)

    return new_x1, new_x2, new_y1, new_y2

