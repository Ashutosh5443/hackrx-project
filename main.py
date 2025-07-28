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
import pypdf
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
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
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Models and services initialized successfully.")
except Exception as e:
    logging.error(f"Initialization failed: {e}")
    raise RuntimeError(f"Could not initialize essential services: {e}")

# --- Helper Functions ---
def download_and_read_pdf(url: str) -> str:
    """Downloads and extracts text from a PDF URL."""
    try:
        logging.info(f"Downloading PDF from: {url}")
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        
        with open("temp_document.pdf", "wb") as f:
            f.write(response.content)
        
        logging.info("PDF downloaded. Extracting text.")
        text = ""
        with open("temp_document.pdf", 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        
        os.remove("temp_document.pdf")
        return text
    except Exception as e:
        logging.error(f"Failed to download or process PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Could not process PDF from URL: {e}")

def get_text_chunks(text: str) -> List[str]:
    """Splits text into paragraphs or large chunks."""
    # Split by double newlines, then filter out empty strings and very short lines
    chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk.strip()) > 10]
    logging.info(f"Split text into {len(chunks)} chunks.")
    return chunks

def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

async def process_single_question(question: str, all_chunks: List[str], chunk_embeddings: np.ndarray) -> str:
    """Processes a single question using a robust keyword + semantic search approach."""
    logging.info(f"Processing question: '{question[:50]}...'")

    # --- Step 1: Keyword Filtering ---
    keywords = [word for word in re.split(r'\W+', question) if len(word) > 3 and word.lower() not in ['what', 'is', 'the', 'does', 'for', 'are', 'this', 'how']]
    
    candidate_chunks = []
    if keywords:
        for chunk in all_chunks:
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', chunk, re.IGNORECASE) for keyword in keywords):
                candidate_chunks.append(chunk)

    if not candidate_chunks:
        logging.warning("No candidate chunks found after keyword filtering. Using all chunks as fallback.")
        candidate_chunks = all_chunks

    logging.info(f"Found {len(candidate_chunks)} candidate chunks after keyword filtering.")

    # --- Step 2: Semantic Ranking ---
    question_embedding = embedding_model.encode(question)
    candidate_indices = [all_chunks.index(c) for c in candidate_chunks]
    candidate_embeddings = chunk_embeddings[candidate_indices]

    similarities = [cosine_similarity(question_embedding, emb) for emb in candidate_embeddings]
    
    top_k_indices = np.argsort(similarities)[-8:][::-1] # Get top 8 for broader context
    top_chunks = [candidate_chunks[i] for i in top_k_indices]
    context = "\n---\n".join(top_chunks)

    # --- Step 3: Answer Generation with a Decisive Prompt ---
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
app = FastAPI(title="HackRx 6.0 Q&A System (Definitive Fix)")

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
    
    # 1. Download and chunk the document text
    document_text = download_and_read_pdf(payload.documents)
    text_chunks = get_text_chunks(document_text)
    
    if not text_chunks:
        raise HTTPException(status_code=500, detail="Failed to extract any text from the document.")

    # 2. Pre-compute embeddings for all chunks (done once per request)
    logging.info("Pre-computing embeddings for all document chunks...")
    chunk_embeddings = embedding_model.encode(text_chunks)
    logging.info("Embeddings computed.")

    # 3. Process all questions in parallel for maximum speed
    logging.info("Processing all questions concurrently...")
    tasks = [process_single_question(q, text_chunks, chunk_embeddings) for q in payload.questions]
    answers = await asyncio.gather(*tasks)

    logging.info("Successfully processed all questions. Returning response.")
    return HackRxRunResponse(answers=answers)
