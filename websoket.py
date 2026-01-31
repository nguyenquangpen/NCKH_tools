import asyncio
import websockets
import json
from config import *

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
    
    async def run_florence(self, callback_func, *args, **kwargs):
        """
        logic: send init --> wait for ready --> send per short data
        """
        await self.ws_connect.send('init_florence')
        response = await self.ws_connect.recv()
        if response != "ready_florence":
            return None
        print("Florence is ready")
        meta_path  = await callback_func(*args, **kwargs)
        if meta_path:
            await self.ws_connect.send("success_florence")
            return meta_path
        return None
            
    async def run_llama(self, prompt_json_path):
        await self.ws_connect.send('init_llama')
        response = await self.ws_connect.recv()
        if response == "ready_llama":
            print("🚀 Server ready for Llama. (Logic will be implemented here)")
            payload = {
                "status": "run_llama",
                "prompt_json_path": prompt_json_path,
            }
            await self.ws_connect.send(json.dumps(payload))
            response = await self.ws_connect.recv()
            res_data = json.loads(response)

            if res_data.get("status") == "completed_llama":
                print("✅ Llama inference completed.")
                await self.ws_connect.send('success_llama')
                return True
            
            elif res_data.get("status") in ["failed_llama", "error_llama", "server_error"]:
                print("❌ Llama failed:", res_data.get("message"))
                return False
        return False

    async def close_ws(self):
        if self.ws_connect:
            await self.ws_connect.close()
            self.ws_connect = None

    
        
        
