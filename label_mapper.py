import numpy as np
import json
import os
import scripy.io


class LabelMapper:
    def __init__(self, mat_file_path):
        self.mat_file_path = mat_file_path
        self.tvsum_data = self.mat_data['tvsum50']

    def _get_video_gt_scores(self, video_id):
        """find and get ground truth scores for a video from mat file"""
        for i in range(self.tvsum_data.shape[1]):
            v_data = self.tvsum_data[0, i]
            v_id_mat = v_data['video'][0]
            if v_id_mat in video_id:
                return v_data['gt_score'].flatten()
        return None
    
    def map_labels(self, metadata_file_path, output_dir="labels"):
        """map ground truth scores to video metadata and save to json files"""
        with open(metadata_file_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        video_id = meta['video_id']
        fps = meta['fps']
        n_frames = meta['n_frames']
        segments = meta['segments']

        full_gt_scores = self._get_video_gt_scores(video_id)
        if full_gt_scores is None:
            print(f"⚠️ Ground truth scores not found for video {video_id}")
            return
        scale_factor = len(full_gt_scores) / n_frames
        segment_labels = []
        for seg in segments:
            s_idx = int(seg['start_frame'] * scale_factor)
            e_idx = int(seg['end_frame'] * scale_factor)

            seg_scores = full_gt_scores[s_idx : e_idx + 1]
            
            if len(seg_scores) > 0:
                avg_score = np.mean(seg_scores)
                norm_score = round((avg_score - 1) / 4, 4)
            else:
                norm_score = 0.0
            
            segment_labels.append({
                "id": seg['id'],
                "gt_score": norm_score
            })

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_path = os.path.join(output_dir, f"{video_id}_labels.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "video_id": video_id,
                "labels": segment_labels
            }, f, indent=2)
            
        print(f"✅ Ground Truth Labels saved: {output_path}")
