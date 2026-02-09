# import h5py
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# def diagnostic_h5(file_path, video_id):
#     with h5py.File(file_path, 'r') as f:
#         data = f[video_id][:].astype(np.float32)
    
#     n_segments = data.shape[0]
#     vectors = data.squeeze(1)
#     sim_matrix = cosine_similarity(vectors)
#     adj_sim = np.mean([sim_matrix[i, i+1] for i in range(n_segments-1)])
#     mean_val = np.mean(vectors)

#     print(f"=== 🩺 CHẨN ĐOÁN: {video_id} ===")
#     print(f"🔗 Similarity liền kề: {adj_sim:.4f}")
    
#     print("\n--- 📋 KẾT LUẬN ---")
    
#     # Logic mới bao phủ toàn bộ các trường hợp
#     if adj_sim > 0.999:
#         print("🚨 CẢNH BÁO: Collapse Risk (Các vector gần như trùng nhau hoàn toàn).")
#     elif 0.995 < adj_sim <= 0.999:
#         print("⚠️ CẢNH BÁO: High Similarity (Hơi cao, nhưng vẫn dùng được cho Self-Attention).")
#     elif 0.90 <= adj_sim <= 0.995:
#         print("✅ KẾT QUẢ TỐT: Healthy Variance (Độ phân hóa lý tưởng).")
#     elif adj_sim < 0.90:
#         print("ℹ️ THÔNG BÁO: Low Similarity (Video có sự thay đổi cảnh rất lớn).")
#     if abs(mean_val) < 0.1:
#         print("✅ Zero-centered: Tốt cho huấn luyện.")

# # --- CHẠY KIỂM TRA ---
# # Thay đường dẫn tới38 file của bạn
# FILE_PATH = "llama_emb/tvsum_sum/user_prompt/user_prompt_pool.h5" 
# VIDEO_ID = "video_38" # Thay ID video bạn muốn check

# diagnostic_h5(FILE_PATH, VIDEO_ID)


#count  video
# if __name__ == "__main__":
#     import os
#     count_path = "prompts"
#     total_files = len([name for name in os.listdir(count_path) if os.path.isfile(os.path.join(count_path, name))])
#     print(f"Tổng số file trong thư mục '{count_path}': {total_files}")

import h5py
file_path = "llama_emb/tvsum_sum/gen/gen_pool.h5"
with h5py.File(file_path, "r") as f:
    keys = list(f.keys())   # lấy danh sách video_id
    print("Số video_id:", len(keys))
    print("Danh sách video_id:")
    print(keys)



# import cv2
# from video_segmenter import VideoSegmenter

# video_path = "dataset/tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-video/video/HT5vyqe0Xaw.mp4"

# # Thử kiểm tra bằng OpenCV thuần trước
# cap = cv2.VideoCapture(video_path)
# print(f"Is Opened: {cap.isOpened()}")
# ret, frame = cap.read()
# print(f"Can read first frame: {ret}")
# cap.release()

# # Thử chạy segmenter
# print("Starting Segmenter...")
# segmenter = VideoSegmenter()
# res = segmenter.detect_scenes(video_path)
# print(f"Result: {res}")


"""
'video_1', 'video_10', 'video_11', 'video_12', 'video_13', 'video_17', 'video_18', 'video_2', 'video_20', 'video_21', 'video_23', 'video_28', 'video_3', 'video_32', 'video_33', 'video_34', 'video_35', 'video_36', 'video_37', 'video_38', 'video_4', 'video_41', 'video_42', 'video_44', 'video_46', 'video_49', 'video_5', 'video_50', 'video_6', 'video_7', 'video_8'
"""