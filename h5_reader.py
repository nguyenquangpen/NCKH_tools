import h5py
import numpy as np

class H5DatasetReader:
    def __init__(self, h5_path):
        self.h5_path = h5_path

    def get_video_change_points(self, video_id):
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if video_id not in f:
                    return None
                cp = f[video_id]['change_points'][()]
                return cp.tolist()
        except Exception as e:
            print(f"Error reading H5 file: {e}")
            return None
# Example usage:
# reader = H5DatasetReader('dataset/TVSum/TVSum/eccv16_dataset_tvsum_google_pool5.h5')
# change_points = reader.get_video_change_points('video_2')
# print(change_points)
