from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from document_processor import DocumentProcessor
import os

app = FastAPI(title="Semantic Search API")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document processor
doc_processor = DocumentProcessor()

# Data directory path from environment variable or default
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")

# Try to load existing index
@app.on_event("startup")
async def startup_load_index():
    try:
        doc_processor.load_index(DATA_DIR)
        print(f"Loaded existing index from {DATA_DIR}")
    except Exception as e:
        print(f"No existing index found in {DATA_DIR}: {e}")
        print("Please build an index before using the search functionality")

class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5

class SearchResult(BaseModel):
    document_id: str
    text: str
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

@app.post("/search", response_model=SearchResponse)
def search(search_query: SearchQuery):
    if doc_processor.faiss_index is None:
        raise HTTPException(status_code=500, detail="Search index not initialized")
    
    results = doc_processor.search(search_query.query, search_query.top_k)
    for r in results:
        r["document_id"] = str(r["document_id"])
    return {"results": results}

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "documents_indexed": len(doc_processor.documents) if doc_processor.documents else 0,
        "data_directory": DATA_DIR
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)