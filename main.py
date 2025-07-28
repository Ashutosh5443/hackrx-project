# main.py
# --- Core Libraries ---
import os
import requests
import logging
import asyncio
import re

# --- Web Framework ---
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List
from fastapi.middleware.cors import CORSMiddleware

# --- Document Processing & AI ---
import fitz  # PyMuPDF
import google.generativeai as genai
import numpy as np

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment Variable Loading ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HACKRX_TOKEN = os.environ.get("HACKRX_TOKEN")

# --- Security ---
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not HACKRX_TOKEN or credentials.credentials != HACKRX_TOKEN:
        logging.warning("Authentication failed: Invalid token provided.")
        raise HTTPException(status_code=403, detail="Invalid authentication credentials")
    return credentials

# --- Pydantic Models ---
class HackRxRunRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document to be processed.")
    questions: List[str] = Field(..., description="A list of questions to be answered based on the document.")

class HackRxRunResponse(BaseModel):
    answers: List[str] = Field(..., description="A list of answers corresponding to the questions.")

# --- Global Initializations ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    logging.info("Models and services initialized successfully.")
except Exception as e:
    logging.error(f"Initialization failed: {e}")
    raise RuntimeError(f"Could not initialize essential services: {e}")

# --- Helper Functions ---
def download_and_read_pdf(url: str) -> str:
    """Downloads and extracts clean text from a PDF URL using PyMuPDF."""
    try:
        logging.info(f"Downloading PDF from: {url}")
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        
        pdf_bytes = response.content
        text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text() + "\n\n" # Add double newline to preserve paragraph breaks
        
        # Clean up residual noise
        text = re.sub(r'\s*\n\s*', '\n', text).strip()
        logging.info(f"Extracted {len(text)} characters from the PDF.")
        return text
    except Exception as e:
        logging.error(f"Failed to download or process PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Could not process PDF from URL: {e}")

def get_text_chunks(text: str) -> List[str]:
    """Splits text into paragraphs or large chunks robustly."""
    chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk.strip()) > 50]
    if not chunks: # Fallback for documents without clear paragraph breaks
        chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
    logging.info(f"Split text into {len(chunks)} chunks.")
    return chunks

def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

async def process_single_question(question: str, all_chunks: List[str], chunk_embeddings: List[List[float]]) -> str:
    """Processes a single question using in-memory semantic search."""
    logging.info(f"Processing question: '{question[:50]}...'")

    # --- Step 1: Semantic Search in Memory ---
    question_embedding_response = await asyncio.to_thread(
        genai.embed_content,
        model='models/text-embedding-004',
        content=question,
        task_type="RETRIEVAL_QUERY"
    )
    question_embedding = question_embedding_response['embedding']

    similarities = [cosine_similarity(question_embedding, emb) for emb in chunk_embeddings]
    
    # Get the top 8 most relevant chunks for context
    top_k_indices = np.argsort(similarities)[-8:][::-1]
    top_chunks = [all_chunks[i] for i in top_k_indices]
    context = "\n---\n".join(top_chunks)

    # --- Step 2: Answer Generation with a Decisive Prompt ---
    prompt = f"""
    You are an AI expert at analyzing policy documents. Your task is to provide a direct, accurate answer to the user's question using ONLY the provided context.

    **Instructions:**
    1.  Your primary goal is to find and state the answer.
    2.  Analyze the 'User's Question' and locate the most relevant information within the 'Context Snippets'.
    3.  Construct a concise answer based *only* on the information you find.
    4.  If the answer is explicitly present, quote it. If it's implied, summarize it.
    5.  DO NOT apologize or state that the answer is not available unless it is absolutely impossible to derive an answer from the text. Be decisive.

    **Context Snippets:**
    {context}

    **User's Question:**
    {question}

    **Answer:**
    """
    try:
        response = await llm_model.generate_content_async(prompt)
        answer = response.text.strip()
        logging.info(f"Generated answer for question: '{question[:50]}...'")
        return answer
    except Exception as e:
        logging.error(f"Error generating answer for '{question[:50]}...': {e}")
        return "An error occurred while generating the answer."

# --- FastAPI Application ---
app = FastAPI(title="HackRx 6.0 Q&A System (High-Accuracy Engine)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "API is running."}

@app.post("/hackrx/run", response_model=HackRxRunResponse)
async def hackrx_run(payload: HackRxRunRequest, _=Depends(get_current_user)):
    """Main endpoint to process a document and questions."""
    logging.info("Received request for /hackrx/run")
    
    document_text = download_and_read_pdf(payload.documents)
    text_chunks = get_text_chunks(document_text)
    
    if not text_chunks:
        raise HTTPException(status_code=500, detail="Failed to extract any text from the document.")

    logging.info("Pre-computing embeddings for all document chunks via Gemini API...")
    embedding_response = await asyncio.to_thread(
        genai.embed_content,
        model='models/text-embedding-004',
        content=text_chunks,
        task_type="RETRIEVAL_DOCUMENT"
    )
    chunk_embeddings = embedding_response['embedding']
    logging.info("Embeddings computed.")

    logging.info("Processing all questions concurrently...")
    tasks = [process_single_question(q, text_chunks, chunk_embeddings) for q in payload.questions]
    answers = await asyncio.gather(*tasks)

    logging.info("Successfully processed all questions. Returning response.")
    return HackRxRunResponse(answers=answers)
