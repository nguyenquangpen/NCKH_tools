import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def diagnostic_h5(file_path, video_id):
    """
    Chẩn đoán sức khỏe Embedding trong file H5.
    """
    with h5py.File(file_path, 'r') as f:
        if video_id not in f:
            print(f"❌ Video ID '{video_id}' không tồn tại trong file.")
            return
        
        # Load dữ liệu và ép kiểu float32 để tính toán chính xác
        data = f[video_id][:].astype(np.float32)
        
    # 1. Kiểm tra cấu trúc (Shape)
    # Kỳ vọng: (N, 1, 4096)
    n_segments = data.shape[0]
    dim = data.shape[2]
    vectors = data.squeeze(1) # Đưa về (N, 4096)

    print(f"=== 🩺 CHẨN ĐOÁN: {video_id} ===")
    print(f"📊 Cấu trúc: {n_segments} segments | Dimension: {dim}")

    # 2. Kiểm tra lỗi dữ liệu (NaN/Inf)
    has_nan = np.isnan(vectors).any()
    has_inf = np.isinf(vectors).any()
    if has_nan or has_inf:
        print(f"🚨 CẢNH BÁO: Dữ liệu chứa NaN={has_nan} hoặc Inf={has_inf}!")
    else:
        print("✅ Dữ liệu sạch (Không có NaN/Inf)")

    # 3. Phân phối thống kê (Statistical Health)
    mean_val = np.mean(vectors)
    std_val = np.std(vectors)
    # Llama-3 qua RMSNorm thường có Mean gần 0 và Std quanh mức 0.3-0.5
    print(f"📈 Mean: {mean_val:.6f} (Kỳ vọng: quanh 0)")
    print(f"📉 Std: {std_val:.6f} (Kỳ vọng: 0.3 - 0.9)")

    # 4. Kiểm tra độ tương quan (Semantic Variance)
    # Tính ma trận Similarity giữa tất cả các cặp shots
    sim_matrix = cosine_similarity(vectors)
    
    # Độ tương đồng trung bình giữa các shot liền kề (Adjacent shots)
    adj_sim = np.mean([sim_matrix[i, i+1] for i in range(n_segments-1)])
    
    # Độ tương đồng thấp nhất và cao nhất trong toàn video
    min_sim = np.min(sim_matrix)
    max_sim_non_diag = np.max(sim_matrix - np.eye(n_segments)) # Bỏ qua đường chéo chính

    print(f"🔗 Similarity liền kề: {adj_sim:.4f} (Kỳ vọng: 0.95 - 0.99)")
    print(f"🌈 Dải Similarity: Min={min_sim:.4f} -> Max={max_sim_non_diag:.4f}")

    # 5. Đánh giá cuối cùng
    print("\n--- 📋 KẾT LUẬN ---")
    status = []
    
    if 0.90 <= adj_sim <= 0.998:
        status.append("✅ Độ phân hóa tốt (Healthy Variance)")
    elif adj_sim > 0.999:
        status.append("⚠️ Cảnh báo: Embedding quá giống nhau (Collapse Risk)")
    else:
        status.append("⚠️ Cảnh báo: Độ tương quan quá thấp (Noise Risk)")

    if abs(mean_val) < 0.1:
        status.append("✅ Zero-centered (Tốt cho Neural Network)")
    
    for s in status:
        print(s)

# --- CHẠY KIỂM TRA ---
# Thay đường dẫn tới file của bạn
FILE_PATH = "llama_emb/tvsum_sum/user_prompt/user_prompt_pool.h5" 
VIDEO_ID = "video_3" # Thay ID video bạn muốn check

diagnostic_h5(FILE_PATH, VIDEO_ID)