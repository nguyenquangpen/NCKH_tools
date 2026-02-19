import json
import os
from video_id_mapper import VideoIdMapper

class PromptGenerator:
    def __init__(self, window_size=8, config_path="prompt_config.md"):
        self.window_size = window_size
        self.config_path = config_path
        self.template = self._load_template()
    
    def _load_template(self):
        """Load prompt template from config file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"❌ can not find file config: {self.config_path}")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _get_window_segments(self, segments, center_idx):
        """get segment around center_idx center"""
        half_w = self.window_size // 2
        start = max(0, center_idx - half_w)
        end = min(len(segments), start + self.window_size)
        if end == len(segments):
            start = max(0, end - self.window_size)
        elif start == 0:
            end = min(len(segments), self.window_size) 
        return segments[start:end]

    def _generate_global_context(self, segments):
        if not segments: return "empty"
        total_shots = len(segments)
        first = segments[0]['caption'][:60]
        mid = segments[total_shots // 2]['caption'][:60]
        last = segments[-1]['caption'][:60]
        return f"This video has {total_shots} shots. It starts with {first}, then moves to {mid}, and concludes with {last}."
    
    def generate_prompts(self, metadata_path, output_dir="prompts"):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        raw_video_id = meta['video_id']
        video_id = VideoIdMapper.get_canonical_id(raw_video_id)
        segments = meta['segments']
        global_ctx = self._generate_global_context(segments)
        video_prompts = []

        for i in range(len(segments)):
            window = self._get_window_segments(segments, i)
            query_content = ""
            for s in window:
                is_target = "[TARGET SHOT] " if s['id'] == segments[i]['id'] else ""
                timestamp = f"({s.get('start_time', '0')}s-{s.get('end_time', '0')}s)"
                query_content += f"{is_target}Shot {s['id']} {timestamp}: {s['caption']}\n"

            full_prompt = self.template \
                .replace("[[global_context]]", global_ctx) \
                .replace("[[query_content]]", query_content)
            
            video_prompts.append({
                "segment_id": segments[i]['id'],
                "prompt": full_prompt
            })
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = os.path.join(output_dir, f"{video_id}_prompts.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                "video_id": video_id, 
                "window_size": self.window_size,
                "prompts": video_prompts
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ Prompts with Sliding Window saved: {out_path}")