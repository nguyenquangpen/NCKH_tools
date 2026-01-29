import asyncio
import cv2
from websoket import Agent

async def main():
    agent = Agent()

    frame = cv2.imread("dataset/tải xuống (2).png")
    result = await agent.send_shot_for_inference(
        shot_id=1,
        frame=frame
    )

    print("Server response:", result)
    await agent.close_ws()

asyncio.run(main())
