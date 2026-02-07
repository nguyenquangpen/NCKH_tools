import base64
import os
import subprocess
import cv2
import shutil
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector 

class VideoSegmenter:
    def __init__(self, threshold = 2.0):
        self.threshold = threshold

    def detect_scenes(self, video_path):
        """ Detect scenes in the video and return change points and segment info."""
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector(
            adaptive_threshold=self.threshold,
            min_content_val=10.0,
            min_scene_len=15
            ))
        print("Detecting scenes...")

        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        change_points = []
        for start, end in scene_list:
            start_frame = start.get_frames()
            end_frame = end.get_frames() - 1
            change_points.append([start_frame, end_frame])

        return {
            "change_points": change_points,
            "n_frames": n_frames,
            "fps": fps
        }

    def extract_single_shot(self, video_path, shot_id, start_frame, end_frame):
        """extract key frame from a single shot"""
        cap = cv2.VideoCapture(video_path)
        mid_frame = (start_frame + end_frame) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        img_b64 = None
        if not ret:
            return None
        
        _, buffer = cv2.imencode('.jpg', frame)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        cap.release()
        
        return {
            "shot_id": shot_id,
            "image_b64": img_b64
        }

    def load_fixed_scenes(self, video_path, manual_change_points):
        """ Load fixed scenes from manual change points """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        return {
            "change_points": manual_change_points,
            "n_frames": n_frames,
            "fps": fps
        }

    """
    Change Points: [[0, 764], [765, 2017], [2018, 2127], [2128, 2611], [2612, 3161], [3162, 4350], [4351, 5009], [5010, 6223], [6224, 6555], [6556, 7584]]
    Number of Frames per Segment: [765, 1253, 110, 484, 550, 1189, 659, 1214, 332, 1029]
    Total Number of Frames: 7585
    Frames per Second (FPS): 59.94005994005994
    """
