from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from prompt import SALES_AGENT_PROMPT
from process import ProductSearch
from policy_process import PolicySearch
from langchain_groq import ChatGroq
import os
import signal
import sys
from dotenv import load_dotenv
import json
import logging
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
llm = ChatGroq(
    temperature=0.3, model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY")
)

# Đường dẫn tới file data.json
DATA_FILE = "data/data.json"

# Khởi tạo các components
product_search = ProductSearch()
policy_search = PolicySearch()


@app.get("/products")
async def get_products():
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            products = json.load(f)
        return products
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"✅ WebSocket connection accepted. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(
                f"WebSocket disconnected. Remaining connections: {len(self.active_connections)}"
            )

    async def send_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
            logger.info("Message sent successfully")
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Xử lý message dựa vào loại
            if "policy" in message_data["message"].lower():
                response = policy_search.get_policy_response(message_data["message"])
            else:
                response = product_search.get_product_response(message_data["message"])

            await manager.send_message(
                json.dumps(
                    {"type": "message", "ai_response": response}, ensure_ascii=False
                ),
                websocket,
            )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Error in websocket endpoint: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    config = uvicorn.Config(app="app:app", host="0.0.0.0", port=8000, reload=True)
    server = uvicorn.Server(config)
    server.run()
