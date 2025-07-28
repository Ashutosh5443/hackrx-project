# main.py
# --- Core Libraries ---
import os
import requests
import logging
from contextlib import contextmanager
import uuid
import asyncio

# --- Web Framework ---
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List
from fastapi.middleware.cors import CORSMiddleware

# --- Document Processing & AI ---
import pypdf
import google.generativeai as genai

# --- Vector Database ---
from pinecone import Pinecone, ServerlessSpec

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment Variable Loading ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HACKRX_TOKEN = os.environ.get("HACKRX_TOKEN")

# --- Security ---
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not HACKRX_TOKEN or credentials.credentials != HACKRX_TOKEN:
        logging.warning("Authentication failed: Invalid token provided.")
        raise HTTPException(status_code=403, detail="Invalid authentication credentials")
    return credentials

# --- Pydantic Models (Matching the Hackathon Spec) ---
class HackRxRunRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document to be processed.")
    questions: List[str] = Field(..., description="A list of questions to be answered based on the document.")

class HackRxRunResponse(BaseModel):
    answers: List[str] = Field(..., description="A list of answers corresponding to the questions.")

# --- Global Initializations ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    EMBEDDING_DIMENSION = 768
    logging.info("Models and services initialized successfully.")
except Exception as e:
    logging.error(f"Initialization failed: {e}")
    raise RuntimeError(f"Could not initialize essential services: {e}")

# --- Helper Functions ---
def download_and_read_pdf(url: str) -> str:
    """Downloads and extracts text from a PDF URL."""
    try:
        logging.info(f"Downloading PDF from: {url}")
        response = requests.get(url, stream=True, timeout=20)
        response.raise_for_status()
        
        temp_pdf_path = "temp_document.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)
        
        logging.info("PDF downloaded. Extracting text.")
        text = ""
        with open(temp_pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        
        os.remove(temp_pdf_path)
        logging.info(f"Extracted {len(text)} characters from the PDF.")
        return text
    except Exception as e:
        logging.error(f"Failed to download or process PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Could not process PDF from URL: {e}")

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """Splits text into coherent, overlapping chunks."""
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    logging.info(f"Split text into {len(chunks)} chunks.")
    return chunks

@contextmanager
def get_pinecone_index():
    """Context manager to create, use, and delete a temporary Pinecone index for each request."""
    index_name = f"hackrx-session-{uuid.uuid4().hex[:8]}"
    try:
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
        
        logging.info(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
        )
        index = pc.Index(index_name)
        yield index
    finally:
        if index_name in pc.list_indexes().names():
            logging.info(f"Deleting Pinecone index '{index_name}' after processing.")
            pc.delete_index(index_name)

async def process_single_question(question: str, index) -> str:
    """Asynchronous function to process one question with a simplified, direct prompt."""
    # 1. Get relevant context from Pinecone
    response = await asyncio.to_thread(
        genai.embed_content,
        model='models/text-embedding-004',
        content=question,
        task_type="RETRIEVAL_QUERY"
    )
    question_embedding = response['embedding']
    
    query_result = await asyncio.to_thread(
        index.query,
        vector=question_embedding,
        top_k=12, 
        include_metadata=True
    )
    context_chunks = [match['metadata']['text'] for match in query_result['matches']]
    context = "\n---\n".join(context_chunks)
    logging.info(f"Retrieved context for question: '{question[:50]}...'")

    # 2. Generate answer with a new, more direct and assertive prompt
    if not context:
        return "The answer to this question is not available in the provided document excerpts."

    prompt = f"""
    You are an AI Information Extractor. Your task is to provide a direct and concise answer to the user's question using ONLY the provided text from a document.

    **Instructions:**
    - Your primary goal is to find the answer. Assume the answer is present in the context.
    - Extract the key information from the **Context Snippets** that directly answers the **User's Question**.
    - Formulate a clear and direct answer.
    - Do not add any information not present in the snippets. Do not apologize or explain. Just provide the answer.

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
app = FastAPI(title="HackRx 6.0 Q&A System (Direct Extraction Engine)")

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
    """
    Main endpoint that processes a document URL and a list of questions.
    """
    logging.info("Received request for /hackrx/run")
    
    document_text = download_and_read_pdf(payload.documents)
    text_chunks = chunk_text(document_text)
    
    if not text_chunks:
        raise HTTPException(status_code=500, detail="Failed to extract any text from the document.")

    with get_pinecone_index() as index:
        logging.info("Generating embeddings for all chunks via Gemini API...")
        response = genai.embed_content(
            model='models/text-embedding-004',
            content=text_chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )
        all_embeddings = response['embedding']
        logging.info("Embeddings generated.")

        vectors_to_upsert = [
            {"id": f"chunk_{i}", "values": embedding, "metadata": {"text": chunk}}
            for i, (chunk, embedding) in enumerate(zip(text_chunks, all_embeddings))
        ]
        
        logging.info(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
        index.upsert(vectors=vectors_to_upsert, batch_size=100)
        logging.info("Successfully upserted vectors.")

        logging.info("Processing all questions concurrently...")
        tasks = [process_single_question(q, index) for q in payload.questions]
        answers = await asyncio.gather(*tasks)

    logging.info("Successfully processed all questions. Returning response.")
    return HackRxRunResponse(answers=answers)
