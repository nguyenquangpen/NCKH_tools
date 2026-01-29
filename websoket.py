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

        if response == "ready_florence":
            print("Florence is ready")
            status = await callback_func(*args, **kwargs)
            
            if status == "success_florence":
                await self.ws_connect.send(status)
                return True
            else:
                return False
            
    async def run_llama(self):
        await self.ws.send('init_llama_3')
        response = await self.ws.recv()
        
        if response == "ready_llama_3":
            print("🚀 Server ready for Llama_3. (Logic will be implemented here)")
            # Thực hiện logic Llama ở đây
            return True
        return False

    async def close_ws(self):
        if self.ws_connect:
            await self.ws_connect.close()
            self.ws_connect = None

    
        
        
