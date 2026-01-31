 
from label_mapper import LabelMapper

LABEL_DIR = "labels"
prompt_json_path = "prompts/video_30_prompts.json"
meta_path = "output/_xMr-HKMfVA.mp4_metadata.json"
mapper = LabelMapper("dataset/tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-matlab/matlab/ydata-tvsum50.mat")
mapper.map_labels(meta_path, output_dir=LABEL_DIR)