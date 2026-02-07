import json
import os
from video_id_mapper import VideoIdMapper

class PromptGenerator:
    def __init__(self, window_size=4, config_path="prompt_config.md"):
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
    
    def _format_few_shot(self):
        examples = [
            {
                "desc": "The image shows a static view of an empty sidewalk with no people, vehicles, or noticeable movement. The scene appears quiet and uneventful, with plain buildings and minimal visual activity.",
                "score": 0
            },
            {
                "desc": "The image displays a street corner with a few parked cars and distant pedestrians. There is limited motion, and the overall scene does not indicate any significant activity.",
                "score": 1
            },
            {
                "desc": "The image shows several people walking along the sidewalk near small shops and traffic lights. Some vehicles are passing by, suggesting moderate urban activity.",
                "score": 2
            },
            {
                "desc": "The image captures a person interacting with a street vendor, while nearby pedestrians observe. Multiple vehicles are moving in the background, indicating an active street scene.",
                "score": 3
            },
            {
                "desc": "The image shows a busy intersection with heavy traffic, multiple pedestrians crossing, street signs, billboards, and visible commercial buildings. The scene appears visually rich and eventful.",
                "score": 4
            },
            {
                "desc": "The image captures a major moment in the video where a public performance or significant gathering is taking place. A large crowd is present, vehicles are stopped, and attention is focused on a central event, making this scene highly important.",
                "score": 5
            }
        ]
        formatted_examples = ""
        for i, ex in enumerate(examples):
            formatted_examples += f"Example {i+1}:\n- Description: {ex['desc']}\n- Score: {ex['score']}\n\n"
        return formatted_examples

    def generate_prompts(self, metadata_path, output_dir="prompts"):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        raw_video_id = meta['video_id']
        video_id = VideoIdMapper.get_canonical_id(raw_video_id)
        segments = meta['segments']
        few_shot_content = self._format_few_shot()
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
                .replace("[[few_shot_examples]]", few_shot_content) \
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