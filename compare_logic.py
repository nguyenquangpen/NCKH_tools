import h5py
import numpy as np
import torch
from scipy.stats import spearmanr
from render_video import VideoSummarizer

# --- CẤU HÌNH ---
CKPT_PATH = "model/ckpt/tvsum_head2_layer3/tvsum/tvsum_split4/best_rho_model/epoch=150-val_sRho=0.336.ckpt"
CONFIG_ARGS = {
    'dataset': 'tvsum', 'model': 'LLMVS', 'tag': 'inference',
    'num_heads': 2, 'num_layers': 3, 'reduced_dim': 2048, 'split_idx': 4
}
DATASET_GT_PATH = "dataset/TVSum/TVSum/eccv16_dataset_tvsum_google_pool5.h5"

def compare_specific_keys(summarizer, h5_mine, h5_author, h5_gt):
    mine_rhos = []
    author_rhos = []
    compared_videos = []

    with h5py.File(h5_mine, 'r') as f_mine, \
         h5py.File(h5_author, 'r') as f_author, \
         h5py.File(h5_gt, 'r') as f_gt:
        
        my_keys = list(f_mine.keys())
        
        print(f"🔍 Tìm thấy {len(my_keys)} video trong file của bạn.")
        print(f"{'Video ID':<15} | {'Rho (Mine)':<12} | {'Rho (Author)':<12} | {'Cải thiện'}")
        print("-" * 70)

        for vid in my_keys:
            # Kiểm tra sự tồn tại của key ở cả 3 file
            if vid not in f_author or vid not in f_gt:
                print(f"ℹ️ {vid:<15} | Bỏ qua: Không tìm thấy key ở file đối chứng.")
                continue

            try:
                # 1. Lấy Ground Truth
                gt_score = f_gt[vid]['gtscore'][()]

                # 2. Tính Rho cho file của BẠN
                pred_mine = summarizer.get_scores(h5_mine, vid)
                if pred_mine is None:
                    continue
                
                pred_m_interp = np.interp(np.linspace(0, 1, len(gt_score)), 
                                        np.linspace(0, 1, len(pred_mine)), pred_mine)
                rho_m, _ = spearmanr(pred_m_interp, gt_score)

                # 3. Tính Rho cho file của TÁC GIẢ
                # Kiểm tra dimension trước khi tính
                if f_author[vid].shape[-1] != 5120:
                    print(f"⚠ {vid:<15} | Bỏ qua: Dimension mismatch ({f_author[vid].shape[-1]} != 5120)")
                    continue
                
                pred_author = summarizer.get_scores(h5_author, vid)
                if pred_author is None:
                    continue

                pred_a_interp = np.interp(np.linspace(0, 1, len(gt_score)), 
                                        np.linspace(0, 1, len(pred_author)), pred_author)
                rho_a, _ = spearmanr(pred_a_interp, gt_score)

                # Kiểm tra giá trị hợp lệ (tránh NaN)
                if not (np.isnan(rho_m) or np.isnan(rho_a)):
                    mine_rhos.append(rho_m)
                    author_rhos.append(rho_a)
                    compared_videos.append(vid)
                    
                    diff = rho_m - rho_a
                    status = "✅ +" if diff > 0 else "❌ "
                    print(f"{vid:<15} | {rho_m:.4f}     | {rho_a:.4f}      | {status}{diff:.4f}")

            except Exception as e:
                print(f"💥 {vid:<15} | Lỗi hệ thống: {e}")

    # --- Tổng kết ---
    print("-" * 70)
    if compared_videos:
        avg_mine = np.mean(mine_rhos)
        avg_author = np.mean(author_rhos)
        std_mine = np.std(mine_rhos)
        std_author = np.std(author_rhos)

        print(f"🏆 KẾT QUẢ TRUNG BÌNH TRÊN {len(compared_videos)} VIDEO HỢP LỆ:")
        print(f"▶️ Rho của Bạn:    {avg_mine:.4f} (±{std_mine:.3f})")
        print(f"▶️ Rho của Tác giả: {avg_author:.4f} (±{std_author:.3f})")
        
        diff_avg = avg_mine - avg_author
        if diff_avg > 0:
            print(f"🚀 Kết quả: Cải thiện được {diff_avg:+.4f} so với tác giả.")
        else:
            print(f"📉 Kết quả: Kém hơn {abs(diff_avg):.4f} so với tác giả.")
    else:
        print("⚠ Không có video nào đủ điều kiện để so sánh.")

if __name__ == "__main__":
    summarizer = VideoSummarizer(CKPT_PATH, CONFIG_ARGS)
    
    # Thay đổi đường dẫn thực tế của bạn ở đây
    PATH_MINE = "llama_emb/tvsum_sum/gen/gen_pool.h5" 
    PATH_AUTHOR = "gen_pool.h5" # Đường dẫn file tác giả bạn để ở đâu đó
    
    compare_specific_keys(summarizer, PATH_MINE, PATH_AUTHOR, DATASET_GT_PATH)