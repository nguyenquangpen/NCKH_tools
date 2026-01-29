from dotenv import load_dotenv
import os

load_dotenv()
WS_ENDPOINT = os.getenv("WS_ENDPOINT")


LLAMA3_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>
{few_shot_examples}

Dữ liệu thực tế cần đánh giá (Query):
{query_content}

Hãy đánh giá mức độ quan trọng cho từng phân cảnh trên theo thang điểm 0.0 - 1.0.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Điểm số: """

SYSTEM_INSTRUCTION = (
    "Bạn là một chuyên gia tóm tắt video. Nhiệm vụ của bạn là đọc mô tả văn bản của các phân cảnh "
    "và đánh giá độ quan trọng của chúng đối với nội dung tổng thể của video."
)

FEW_SHOT_EXAMPLES = [
    {
        "caption": "Máy quay lướt qua một đám đông đang đứng chờ bình thường, không có hoạt động gì đặc biệt.",
        "score": 0.15
    },
    {
        "caption": "Vận động viên thực hiện cú nhảy vọt qua rào cản trong tiếng reo hò của khán giả.",
        "score": 0.88
    }
]

SLIDING_WINDOW_SIZE = 5 