# api.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict
import uuid
from rag import RAGAssistant
import uvicorn
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from config import SEARCH_SERVICE_ENDPOINT, SEARCH_SERVICE_KEY

app = FastAPI()
rag_assistants: Dict[str, RAGAssistant] = {}

class Question(BaseModel):
    text: str
    user_id: str

# Helper function to check if index exists
async def index_exists(user_id: str):
    try:
        index_name = f"inventory-{user_id}"
        index_client = SearchIndexClient(
            endpoint=SEARCH_SERVICE_ENDPOINT,
            credential=AzureKeyCredential(SEARCH_SERVICE_KEY)
        )
        indexes = list(index_client.list_index_names())
        return index_name in indexes
    except Exception as e:
        print(f"Error checking index existence: {str(e)}")
        return False

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
            # Check if index exists before initializing
            if await index_exists(question.user_id):
                rag_assistant = RAGAssistant(question.user_id)
                # Make sure search client is initialized
                if rag_assistant.vector_store.search_client is None:
                    await rag_assistant.vector_store.connect_to_index()
                rag_assistants[question.user_id] = rag_assistant
            else:
                await initialize_user_rag(question.user_id)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"User {question.user_id} not found or initialization failed: {str(e)}")
    
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