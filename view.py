import cv2
import json
import base64
import h5py
import os
import asyncio
import glob
from video_segmenter import *
from websoket import WebSocketClient
from video_segmenter import VideoSegmenter
from label_mapper import LabelMapper
from video_id_mapper import VideoIdMapper
from prompt_generator import PromptGenerator
from h5_reader import H5DatasetReader
from render_video import VideoSummarizer

OUTPUT_DIR = "output"
PROMPT_DIR = "prompts"
LABEL_DIR = "labels"
DATASET_H5_PATH = "dataset/TVSum/TVSum/eccv16_dataset_tvsum_google_pool5.h5"
LABELMAPPER = "dataset/tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-matlab/matlab/ydata-tvsum50.mat"
LLAMA_H5_PATH = "llama_emb/tvsum_sum/gen/gen_pool.h5"
CKPT_PATH = "model/ckpt/tvsum_head2_layer3/tvsum/tvsum_split4/best_rho_model/epoch=150-val_sRho=0.336.ckpt"
CONFIG_ARGS = {
    'dataset': 'tvsum',
    'num_heads': 2,
    'num_layers': 3,
    'reduced_dim': 2048,
    'split_idx': 4
}

for d in [OUTPUT_DIR, PROMPT_DIR, LABEL_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

def save_video_metadata(video_id, result_meta, all_segments):
    """save information about video and segments to json file"""
    output_data = {
        "video_id": video_id,
        "fps": result_meta["fps"],
        "n_frames": result_meta["n_frames"],
        "segments": all_segments
    }
    file_path = os.path.join(OUTPUT_DIR, f"{video_id}_metadata.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Metadata & Captions saved: {file_path}")
    return file_path


def is_video_h5(video_id):
    """Check if video ID exists in H5 dataset."""
    if not os.path.exists(LLAMA_H5_PATH):
        return False
    try:
        with h5py.File(LLAMA_H5_PATH, 'r') as f:
            return video_id in f
    except Exception as e:
        print(f"Error accessing H5 file: {e}")
        return False

async def _handle_dataset_florence_logic(video_path, ws_client, video_id):
    """Handle dataset Florence logic."""
    reader = H5DatasetReader(DATASET_H5_PATH)
    change_points = reader.get_video_change_points(video_id)
    if not change_points:
        print(f"❌ No change points found in H5 for {video_id}")
        return None
    segmenter = VideoSegmenter()
    result = segmenter.load_fixed_scenes(video_path, change_points)
    fps = result.get("fps", 1.0)
    n_frames = result.get("n_frames", 0)
    all_segments = []

    try:
        for i, (start, end) in enumerate(change_points):
            short_data = segmenter.extract_single_shot(video_path, i, start, end)
            if short_data:
                payload = {
                    "status": "run_florence",
                    "shot_id": short_data["shot_id"],
                    "image_b64": short_data["image_b64"]
                }
                print(f"🚀 Sending shot {i} for inference...")
                await ws_client.send_data(payload)
                response = await ws_client.ws_connect.recv()
                res_json = json.loads(response)
                if res_json.get("status") == "completed":
                    segment_info = {
                        "id": i,
                        "start_frame": start,
                        "end_frame": end,
                        "start_time": round(start / fps, 2),
                        "end_time": round(end / fps, 2),
                        "caption": res_json.get("caption", "")
                    }
                    all_segments.append(segment_info)
                else:
                    print(f"⚠️ Inference failed for shot {i}")
                    return None
        if all_segments:
            result_meta = {"fps": fps, "n_frames": n_frames}
            meta_path = save_video_metadata(video_id, result_meta, all_segments)
            return meta_path
        return None
    except Exception as e:
        print("❌ Error during Florence handling:", e)
        return None


async def _handle_florence_logic(video_path, ws_client, video_id):
    """Handle Florence inference logic."""
    segmenter = VideoSegmenter()
    result = segmenter.detect_scenes(video_path)
    change_points = result["change_points"]
    fps = result.get("fps", 1.0)
    n_frames = result.get("n_frames", 0)

    all_segments = []

    try:
        for i, (start, end) in enumerate(change_points):
            short_data = segmenter.extract_single_shot(video_path, i, start, end)
            if short_data:
                payload = {
                    "status": "run_florence",
                    "shot_id": short_data["shot_id"],
                    "image_b64": short_data["image_b64"]
                }
                print(f"🚀 Sending shot {i} for inference...")
                await ws_client.send_data(payload)

                print(f"⏳ Waiting for response for shot {i}...")
                response = await ws_client.ws_connect.recv()

                print(f"📨 Received response for shot {i}")
                res_json = json.loads(response)
                if res_json.get("status") == "completed":
                    segment_info = {
                        "id": i,
                        "start_frame": start,
                        "end_frame": end,
                        "start_time": round(start / fps, 2),
                        "end_time": round(end / fps, 2),
                        "caption": res_json.get("caption", "")
                    }
                    all_segments.append(segment_info)
                else:
                    print(f"⚠️ Inference failed for shot {i}")
                    return None
        if all_segments:
            result_meta = {"fps": fps, "n_frames": n_frames}
            meta_path = save_video_metadata(video_id, result_meta, all_segments)
            return meta_path
            
        return None
    except Exception as e:
        print("❌ Error during Florence handling:", e)
        return None
    

async def process_florence(video_path, ws_client, use_dataset_mode=True):
    try:
        video_filename = os.path.basename(video_path)
        canonical_id = VideoIdMapper.get_canonical_id(video_filename)
        meta_path = os.path.join(OUTPUT_DIR, f"{canonical_id}_metadata.json")
        prompt_path = os.path.join(PROMPT_DIR, f"{canonical_id}_prompts.json")
        if os.path.exists(meta_path) and os.path.exists(prompt_path):
            return {
                "status": "success",
                "canonical_id": canonical_id,
                "meta_path": meta_path,
                "prompt_path": prompt_path,
                "skipped": True
            }
        logic_func = _handle_dataset_florence_logic if use_dataset_mode else _handle_florence_logic
        meta_path = await ws_client.run_florence(
            logic_func,
            video_path,
            ws_client,
            canonical_id
        )

        if not meta_path:
            return {"status": "error", "msg": "florence"}
        prompt_gen = PromptGenerator()
        prompt_gen.generate_prompts(meta_path, output_dir=PROMPT_DIR)
        prompt_path = os.path.join(
            PROMPT_DIR,
            f"{canonical_id}_prompts.json"
        )

        return {
            "status": "success",
            "canonical_id": canonical_id,
            "meta_path": meta_path,
            "prompt_path": prompt_path
        }
    except Exception as e:
        return {"status": "error", "msg": str(e)}


async def process_llama(video_info, ws_client, traning_mode=True):
    if video_info["status"] != "success":
        return "skip"
    try:
        cid = video_info["canonical_id"]
        if is_video_h5(cid):
            return "skip"

        label_path = os.path.join(LABEL_DIR, f"{cid}_labels.json")
        if not traning_mode and os.path.exists(label_path):
            return "skip" 
        
        llama_success = await ws_client.run_llama(
            video_info["prompt_path"],
            cid
        )

        if not llama_success:
            return "failure_llama"
        
        if not traning_mode:
            mapper = LabelMapper(LABELMAPPER)
            mapper.map_labels(
                video_info["meta_path"],
                output_dir=LABEL_DIR
            )
        return "success"
    except Exception as e:
        print(f"❌ Error during Llama processing: {e}")
        return "failure_pipeline"


def get_completed_florence_tasks():
    """Retrieve previously completed Florence tasks from output directory."""
    tasks = []
    prompt_file = glob.glob(os.path.join(PROMPT_DIR, "*_prompts.json"))
    for p_path in prompt_file:
        filename = os.path.basename(p_path)
        canonical_id = filename.replace("_prompts.json", "")
        meta_path = os.path.join(OUTPUT_DIR, f"{canonical_id}_metadata.json")
        if os.path.exists(meta_path):
            tasks.append({
                "status": "success",
                "canonical_id": canonical_id,
                "meta_path": meta_path,
                "prompt_path": p_path
            })
    return tasks


async def run_phase_florence(video_paths):
    ws_client = WebSocketClient()
    if not await ws_client.connect_ws():
        print("Cannot connect to WS for Phase 1")
        return []
    successful_tasks = []
    try:
        if not await ws_client.prepare_florence():
            print("Florence Prep Failed")
            return []
        
        for i, v_path in enumerate(video_paths):
            v_name = os.path.basename(v_path)
            cid = VideoIdMapper.get_canonical_id(v_name)
            if os.path.exists(os.path.join(PROMPT_DIR, f"{cid}_prompts.json")):
                continue
            print(f"📽️ Processing Florence [{i+1}/{len(video_paths)}]: {os.path.basename(v_path)}")
            res = await process_florence(v_path, ws_client)
            if res.get("status") == "success":
                successful_tasks.append(res)
        await ws_client.finish_florence()
    finally:
        await ws_client.close_ws()
    return successful_tasks


async def run_phase_llama(tasks=None):
    if tasks is None:
        tasks = get_completed_florence_tasks()

    if not tasks:
        print("No tasks to process in Phase 2")
        return
    
    ws_client = WebSocketClient()
    if not await ws_client.connect_ws():
        print("Cannot connect to WS for Phase 2")
        return
    try:
        if not await ws_client.prepare_llama():
            print("Llama preparation failed")
            return
        for i, info in enumerate(tasks):
            cid = info["canonical_id"]
            print(f"📽️ Processing Llama [{i+1}/{len(tasks)}]: {cid}")
            res = await process_llama(info, ws_client)
            if res == "skip":
                print(f"⏭️ Skipped Llama for {cid}, already processed.")
            elif res == "success":
                print(f"✅ Finished Llama [{i+1}/{len(tasks)}]: {cid}")
            else:
                print(f"❌ Failed Llama [{i+1}/{len(tasks)}]: {cid} - Status: {res}")
        await ws_client.finish_llama()

    finally:
        await ws_client.close_ws()


async def run_phase_render(video_path):
    summarizer = VideoSummarizer(CKPT_PATH, CONFIG_ARGS)
    for v_path in video_path:
        cid = VideoIdMapper.get_canonical_id(os.path.basename(v_path))
        ouput_video = os.path.join(OUTPUT_DIR, f"{cid}_summary.mp4")
        if os.path.exists(ouput_video):
            print(f"⏭️ Skipping rendering for {cid}, summary already exists.")
            continue

        print(f"📽️ Rendering summary for: {cid}")
        scores = summarizer.get_scores(LLAMA_H5_PATH, cid)
        if scores is None:
            selected_idx, cps = summarizer.select_shots(cid, scores)
            summarizer.render(v_path, selected_idx, cps, ouput_video)
            print(f"✅ Summary video saved: {ouput_video}")


if __name__ == "__main__":
    import glob
    RUN_PHASE_1 = False
    RUN_PHASE_2 = True
    RUN_PHASE_3 = False

    if RUN_PHASE_1:
        VIDEO_DIR = "dataset/tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-video/video" 
        video_paths = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"), recursive=True)
        print(f"--- STARTING PHASE 1: FLORENCE-2 FOR ALL VIDEOS ---")
        asyncio.run(run_phase_florence(video_paths))
        
    if RUN_PHASE_2:
        print(f"\n--- STARTING PHASE 2: LLAMA-3 FOR ALL SUCCESSFUL VIDEOS ---")
        asyncio.run(run_phase_llama())
    
    if RUN_PHASE_3:
        print(f"\n--- STARTING PHASE 3: RENDERING SUMMARY VIDEOS ---")
        asyncio.run(run_phase_render(video_paths))