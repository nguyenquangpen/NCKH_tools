# def check_total_segments(h5_path):
#     import h5py

#     with h5py.File(h5_path, 'r') as f:
#         print(f"{'Video ID':<15} | {'#Segments (shots)':<20}")
#         print("-" * 40)

#         for vid in f.keys():
#             try:
#                 # Cách chuẩn: dùng change_points nếu có
#                 if 'change_points' in f[vid]:
#                     num_segments = len(f[vid]['change_points'])
                
#                 # Fallback: dùng n_frames hoặc features length
#                 elif 'features' in f[vid]:
#                     num_segments = f[vid]['features'].shape[0]
                
#                 else:
#                     num_segments = "Unknown"

#                 print(f"{vid:<15} | {num_segments}")
            
#             except Exception as e:
#                 print(f"{vid:<15} | Error: {e}")

# if __name__ == "__main__":
    
#     # Thay đổi đường dẫn thực tế của bạn ở đây
#     PATH_MINE = "llama_emb/tvsum_sum/gen/gen_pool.h5" 
#     PATH_AUTHOR = "gen_pool.h5" # Đường dẫn file tác giả bạn để ở đâu đó
    
#     check_total_segments(PATH_MINE)

from rembg import remove
from PIL import Image

input_path = 'preview/Screenshot 2026-04-08 170659.png'
output_path = 'chart_no_bg.png'

input_image = Image.open(input_path)
output_image = remove(input_image)
output_image.save(output_path)