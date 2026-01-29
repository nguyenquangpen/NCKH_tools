import json
import os
from prompt_config import PROMPT_TEMPLATE, SYSTEM_INSTRUCTION, FEW_SHOT_EXAMPLES, WINDOW_SIZE

class PromptGenerator:
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size

    def _get_window_segments(self, segments, center_idx):
        """Lấy các segment xung quanh vị trí center_idx"""
        half_w = self.window_size // 2
        start = max(0, center_idx - half_w)
        end = min(len(segments), start + self.window_size)
        
        # Điều chỉnh lại start nếu chạm cuối danh sách để đủ window_size
        if end == len(segments):
            start = max(0, end - self.window_size)
            
        return segments[start:end]

    def generate_prompts(self, metadata_path, output_dir="prompts"):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        video_id = meta['video_id']
        segments = meta['segments']
        video_prompts = []

        for i in range(len(segments)):
            # 1. Tạo cửa sổ trượt
            window = self._get_window_segments(segments, i)
            
            # 2. Định dạng nội dung Query
            query_content = ""
            for s in window:
                prefix = "[Mục tiêu] " if s['id'] == segments[i]['id'] else ""
                query_content += f"{prefix}Phân cảnh {s['id']}: {s['caption']}\n"

            # 3. Đóng gói vào Template
            full_prompt = PROMPT_TEMPLATE.format(
                system_instruction=SYSTEM_INSTRUCTION,
                few_shot_examples=FEW_SHOT_EXAMPLES,
                query_content=query_content
            )

            video_prompts.append({
                "segment_id": segments[i]['id'],
                "prompt": full_prompt
            })

        # 4. Lưu kết quả
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        out_path = os.path.join(output_dir, f"{video_id}_prompts.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({"video_id": video_id, "prompts": video_prompts}, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Prompts with Sliding Window saved: {out_path}")

if __name__ == "__main__":
    generator = PromptGenerator()
    generator.generate_prompts("output/videoplayback (9).mp4_metadata.json")