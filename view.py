import cv2
import json
import base64
import os
import asyncio
from video_segmenter import *
from websoket import WebSocketClient

OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_video_metadata(video_path, result_meta, all_segments):
    """save information about video and segments to json file"""
    video_name = os.path.basename(video_path)
    output_data = {
        "video_id": video_name,
        "fps": result_meta["fps"],
        "n_frames": result_meta["n_frames"],
        "segments": all_segments
    }
    file_path = os.path.join(OUTPUT_DIR, f"{video_name}_metadata.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Metadata & Captions saved: {file_path}")


async def _handle_florence_logic(video_path, ws_client):
    """Handle Florence inference logic."""
    segmenter = VideoSegmenter()
    result = segmenter.detect_scenes(video_path)
    change_points = result["change_points"]
    fps = result.get("fps", 1.0)

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
                    return "failure_florence"
        if all_segments:
            save_video_metadata(video_path, all_segments)
            
        return "success_florence"
    except Exception as e:
        print("❌ Error during Florence handling:", e)
        return "failure_florence"
    
async def main_process(video_path):
    ws_client = WebSocketClient()
    if not await ws_client.connect_ws():
        return
    
    try:
        success = await ws_client.run_florence(_handle_florence_logic, video_path, ws_client)
        return "sccess_florence" if success else "failure_florence"
    finally:
        await ws_client.close_ws()

def florence_callback(video_path):
    """Entry point for Florence processing."""
    return asyncio.run(main_process(video_path))

if __name__ == "__main__":
    # Test thử
    video_path = "dataset/videoplayback (9).mp4"
    result = florence_callback(video_path)
    print(f"Final Status: {result}")