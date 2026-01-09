import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

# Load env
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("GENAI_API_KEY not set")

# Gemini client
client = genai.Client(api_key=API_KEY)

# FastAPI app
app = FastAPI(title="Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    ok: bool
    query: str
    response: str

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Search API is running",
        "status": "ok",
        "endpoints": {
            "search": "/search (POST)",
            "docs": "/docs",
            "health": "/health"
        }
    }

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Search endpoint
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    try:
        system_prompt = """
You are a helpful AI assistant.

Rules:
- Answer naturally like ChatGPT or Gemini.
- The response should depend on the user's question.
- If the user asks for explanation, explain.
- If the user asks for advantages/disadvantages, provide them.
- If the user asks for code, provide code.
- If the user asks a simple question, give a simple answer.
- Do NOT force unnecessary structure.
"""

        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-3-flash-preview",
            contents=[
                system_prompt,
                request.query
            ],
            config={
                "temperature": 0.6,
                "max_output_tokens": 800
            }
        )

        return SearchResponse(
            ok=True,
            query=request.query,
            response=response.text
        )

    except Exception as e:
        return SearchResponse(
            ok=False,
            query=request.query,
            response=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)