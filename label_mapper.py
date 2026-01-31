import numpy as np
import json
import os
import h5py

class LabelMapper:
    def __init__(self, mat_file_path):
        self.mat_file_path = mat_file_path
        self.mat_data = h5py.File(self.mat_file_path, 'r')
        self.tvsum_data = self.mat_data['tvsum50']

    def _matlab_str_to_py(self, mat_arr):
        """Convert MATLAB string from h5py to Python string"""
        obj = self.mat_data[mat_arr]
        return ''.join(chr(int(c)) for c in np.array(obj).flatten())

    def _get_video_gt_scores(self, video_id):
        video_refs = self.tvsum_data['video'][:].flatten()
        gt_refs = self.tvsum_data['gt_score'][:].flatten()
        meta_id = video_id.replace(".mp4", "").strip()

        for i, v_ref in enumerate(video_refs):
            actual = self._matlab_str_to_py(v_ref)
            if actual == meta_id:
                gt_ref = gt_refs[i]
                scores = np.array(self.mat_data[gt_ref]).flatten()
                return scores
        return None
    
    def map_labels(self, metadata_file_path, output_dir="labels"):
        """map ground truth scores to video metadata and save to json files"""
        with open(metadata_file_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        video_id = meta['video_id']
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

            s_idx = max(0, s_idx)
            e_idx = min(len(full_gt_scores) - 1, e_idx)
            seg_scores = full_gt_scores[s_idx : e_idx + 1]
            
            if len(seg_scores) > 0:
                avg_score = np.mean(seg_scores)
                final_score = int(round(float(avg_score)))
                final_score = max(1, min(5, final_score))
            else:
                final_score = 1

            segment_labels.append({
                "id": seg['id'],
                "gt_score": final_score
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
