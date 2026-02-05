import h5py
import numpy as np
import os

def save_llama_embeddings(video_id, x1, x2, base_dir="llama_emb/tvsum_sum"):
    """save embeddings to h5 file"""
    try:
        user_path = os.path.join(base_dir, "user_prompt", "user_prompt_pool.h5")
        gen_path = os.path.join(base_dir, "gen", "gen_pool.h5")

        os.makedirs(os.path.dirname(user_path), exist_ok=True)
        os.makedirs(os.path.dirname(gen_path), exist_ok=True)

        def append_to_h5(file_path, v_id, embedding_data):
            data_to_append = np.array(embedding_data)[np.newaxis, np.newaxis, :].astype(np.float16)
            
            with h5py.File(file_path, 'a') as f:
                if v_id not in f:
                    f.create_dataset(
                        v_id, 
                        data=data_to_append,
                        maxshape=(None, 1, data_to_append.shape[2]),
                        chunks=True
                    )
                else:
                    dataset = f[v_id]
                    dataset.resize((dataset.shape[0] + 1), axis=0)
                    dataset[-1:] = data_to_append

        append_to_h5(user_path, video_id, x1)
        append_to_h5(gen_path, video_id, x2)
        return True
    except Exception as e:
        print(f"❌ Error saving Llama embeddings for {video_id}: {e}")
        return False