import os
import torch
import h5py
import numpy as np
import argparse
import json
from model.networks.model import LLMVS
from model.utils.configs import Config
from model.utils.knapsack_implementation import knapSack 
from moviepy.editor import VideoFileClip, concatenate_videoclips


class VideoSummarizer:
    def __init__(self, model_path, config_dict):
        self.config = Config(**config_dict)
        self.model = LLMVS.load_from_checkpoint(model_path, config=self.config)
        self.model.cuda().eval()

    def get_scores(self, llama_h5_path, video_id):
        """Get importance scores from Llama embeddings stored in H5 file"""
        with h5py.File(llama_h5_path, 'r') as f:
            if video_id not in f:
                print("cannot find video in llama h5")
                return None
            emb = torch.from_numpy(f[video_id][()]).float().cuda()
            if len(emb.shape) == 2:
                emb = emb.unsqueeze(0)
        with torch.no_grad():
            logits = self.model(emb, mask=None)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        return probs
    
    def select_shots(self, video_id, probs, ratio=0.15):
        meta_path = f"output/{video_id}_metadata.json"
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        cps = [[s['start_frame'], s['end_frame']] for s in meta['segments']]
        n_frames = meta['n_frames']
        print(f"Loaded metadata from {meta_path}")
        
        short_lengths = [cp[1] - cp[0] for cp in cps]
        if len(probs) != len(short_lengths):
            probs = np.interp(np.linspace(0, 1, len(short_lengths)), np.linspace(0, 1, len(probs)), probs)
        max_len = int(n_frames * ratio)
        selected = knapSack(int(max_len), short_lengths, probs, len(probs))
        return selected, cps
    
    def render(self, input_mp4, selected_indices, cps, output_path):
        print("Rendering summary video...")
        video = VideoFileClip(input_mp4)
        fps = video.fps
        clips = []
        for idx in selected_indices:
            start_f, end_f = cps[idx]
            start_t, end_t = start_f / fps, end_f / fps
            
            clip = video.subclip(start_t, end_t)
            clips.append(clip)

        final_video = concatenate_videoclips(clips)
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print("Summary video saved to:", output_path)
