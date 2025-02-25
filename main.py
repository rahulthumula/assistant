# api.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict
import uuid
from rag import RAGAssistant
import uvicorn

app = FastAPI()
rag_assistants: Dict[str, RAGAssistant] = {}

class Question(BaseModel):
    text: str
    user_id: str

@app.post("/initialize/{user_id}")
async def initialize_user_rag(user_id: str):
    try:
        rag_assistant = RAGAssistant(user_id)
        await rag_assistant.initialize()
        rag_assistants[user_id] = rag_assistant
        return {"message": f"RAG system initialized for user {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(question: Question):
    if question.user_id not in rag_assistants:
        try:
            await initialize_user_rag(question.user_id)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"User {question.user_id} not found or initialization failed")
    
    try:
        rag_assistant = rag_assistants[question.user_id]
        response = await rag_assistant.query(question.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh/{user_id}")
async def refresh_user_index(user_id: str):
    try:
        if user_id in rag_assistants:
            await rag_assistants[user_id].index_user_documents()
            return {"message": f"Index refreshed for user {user_id}"}
        else:
            raise HTTPException(status_code=404, detail=f"User {user_id} not initialized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)