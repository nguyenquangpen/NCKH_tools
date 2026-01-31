import json
import os
from video_id_mapper import VideoIdMapper
# from prompt_config import PROMPT_TEMPLATE, SYSTEM_INSTRUCTION, FEW_SHOT_EXAMPLES, WINDOW_SIZE

class PromptGenerator:
    def __init__(self, window_size=5, config_path="prompt_config.md"):
        self.window_size = window_size
        self.config_path = config_path
        self.template = self._load_template()
    
    def _load_template(self):
        """Load prompt template from config file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file cấu hình: {self.config_path}")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _get_window_segments(self, segments, center_idx):
        """Lấy các segment xung quanh vị trí center_idx"""
        half_w = self.window_size // 2
        start = max(0, center_idx - half_w)
        end = min(len(segments), start + self.window_size)
        
        if end == len(segments):
            start = max(0, end - self.window_size)
        elif start == 0:
            end = min(len(segments), self.window_size) 
        return segments[start:end]
    
    def _format_few_shot(self):
        examples = [
            {
                "desc": "Cảnh quay cận cảnh một chiếc lốp xe cũ bị hỏng nằm bên đường, không có chuyển động.",
                "score": "1"
            },
            {
                "desc": "Người thợ bắt đầu dùng kích nâng xe lên, hành động dứt khoát và rõ ràng.",
                "score": "8"
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
        video_prompts = []

        for i in range(len(segments)):
            window = self._get_window_segments(segments, i)
            query_content = ""
            for s in window:
                is_target = "[TARGET SHOT] " if s['id'] == segments[i]['id'] else ""
                query_content += f"{is_target}Shot {s['id']}: {s['caption']}\n"

            full_prompt = self.template.format(
                few_shot_examples=few_shot_content,
                query_content=query_content
            )
            
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

# if __name__ == "__main__":
#     generator = PromptGenerator()
#     generator.generate_prompts("output/videoplayback (9).mp4_metadata.json")