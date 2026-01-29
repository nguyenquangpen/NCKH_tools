import base64
import os
import subprocess
import cv2
import shutil
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector 

class VideoSegmenter:
    def __init__(self, threshold = 2.7):
        self.threshold = threshold

    def detect_scenes(self, video_path):
        """ Detect scenes in the video and return change points and segment info."""
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=self.threshold))
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


    """
    Change Points: [[0, 764], [765, 2017], [2018, 2127], [2128, 2611], [2612, 3161], [3162, 4350], [4351, 5009], [5010, 6223], [6224, 6555], [6556, 7584]]
    Number of Frames per Segment: [765, 1253, 110, 484, 550, 1189, 659, 1214, 332, 1029]
    Total Number of Frames: 7585
    Frames per Second (FPS): 59.94005994005994
    """

   
# def export_scene_clips_ffmpeg(video_path, change_points, fps, out_dir, num_cuts=2):
#     """ preview scene clips using ffmpeg based on change points."""
#     if os.path.exists(out_dir):
#         shutil.rmtree(out_dir)
#     os.makedirs(out_dir, exist_ok=True)
#     for i, (start, end) in enumerate(change_points[:num_cuts]):
#         start_time = start / fps
#         duration = (end - start + 1) / fps
#         out_path = os.path.join(out_dir, f"scene_{i}.mp4")
#         cmd = [
#             "ffmpeg",
#             "-y",
#             "-ss", f"{start_time:.4f}",
#             "-i", video_path,           
#             "-t", f"{duration:.4f}",    
#             "-c:v", "libx264",         
#             "-preset", "ultrafast",   
#             "-crf", "18",               
#             out_path
#         ]
#         subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         print(f"Exported accurate scene clip to {out_path}")
