import asyncio
import websockets
import json
from config import *
from embedding_storage import save_llama_embeddings

class WebSocketClient:
    def __init__(self):
        self.ws_url = f"{WS_ENDPOINT}/ws/agent"
        self.ws_connect = None
    
    async def connect_ws(self):
        try:
            print("Connecting to:", self.ws_url)
            self.ws_connect = await websockets.connect(
                self.ws_url,
                ping_interval=None,
                ping_timeout=None,
                max_size=None
            )
            print("✅ Connected to Server")
            return True
        except Exception as e:
            print("❌ Connection failed:", e)
            return False
    
    async def send_data(self, data):
        """send data to server"""
        if self.ws_connect:
            await self.ws_connect.send(json.dumps(data))

    # --- PHASE PREPARATION ---
    async def prepare_florence(self):
        await self.ws_connect.send('init_florence')
        response = await self.ws_connect.recv()
        if response == "ready_florence":
            print("Florence is ready")
            return True
        return False
    
    async def prepare_llama(self):
        await self.ws_connect.send('init_llama')
        response = await self.ws_connect.recv()
        if response == "ready_llama":
            print("Llama is ready")
            return True
        return False
    
    # --- PHASE TERMINATION  ---
    async def finish_florence(self):
        await self.ws_connect.send('success_florence')
        response = await self.ws_connect.recv()
        print(f"🧹 Florence Unload Status: {response}")
        return True

    async def finish_llama(self):
        await self.ws_connect.send('success_llama')
        response = await self.ws_connect.recv()
        print(f"🧹 Llama Unload Status: {response}")
        return True
    
    # --- CORE PROCESSING ---
    async def run_florence(self, callback_func, *args, **kwargs):
        """
        logic: send init --> wait for ready --> send per short data
        """
        print("Florence is ready")
        meta_path  = await callback_func(*args, **kwargs)
        return meta_path
            
    async def run_llama(self, prompt_json_path, video_id):
        with open(prompt_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        prompts = data.get("prompts", [])
        for seg in prompts:
            payload = {
                "status": "run_llama",
                "segment_data": seg
            }
            await self.ws_connect.send(json.dumps(payload))
            response = await self.ws_connect.recv()
            res_data = json.loads(response)
            if res_data.get("status") == "completed_llama":
                embedding_status = save_llama_embeddings(
                    video_id,
                    res_data.get("x1"),
                    res_data.get("x2")
                )
                if not embedding_status:
                    print(f"❌ Saving embeddings failed for segment {seg.get('segment_id')}")
                    return False
                print(f"✅ Llama completed for segment {seg.get('segment_id')}")
            else:
                print(f"❌ Llama failed for segment {seg.get('segment_id')}: {res_data.get('message')}")
                return False
        return True

    async def close_ws(self):
        if self.ws_connect:
            await self.ws_connect.close()
            self.ws_connect = None

    
        
        
