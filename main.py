# main.py
# --- Core Libraries ---
import os
import requests
import logging
from contextlib import contextmanager

# --- Web Framework ---
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List

# --- Document Processing & AI ---
import pypdf
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --- Vector Database ---
from pinecone import Pinecone, ServerlessSpec

# --- Logging Configuration ---
# Configure logging to provide detailed output for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment Variable Loading ---
# Load API keys and other configurations from the environment.
# In a local setup, you would use a .env file. In production (like Railway),
# these are set in the platform's variable management system.
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT") # e.g., 'us-east-1'
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HACKRX_TOKEN = os.environ.get("HACKRX_TOKEN")

# --- Security and Authentication ---
# Sets up a bearer token security scheme. The API will require an
# "Authorization: Bearer <token>" header for protected endpoints.
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dependency function to validate the bearer token.
    It checks if the provided token matches the one specified in the environment.
    Raises an HTTPException (403 Forbidden) if the token is invalid.
    """
    if not HACKRX_TOKEN or credentials.credentials != HACKRX_TOKEN:
        logging.warning("Authentication failed: Invalid token provided.")
        raise HTTPException(
            status_code=403,
            detail="Invalid authentication credentials",
        )
    return credentials

# --- Pydantic Models for API Data Validation ---
# These models define the expected structure of the JSON data for
# API requests and responses. FastAPI uses them to validate incoming data
# and serialize outgoing data.

class HackRxRunRequest(BaseModel):
    """Defines the structure for the /hackrx/run request body."""
    documents: str = Field(..., description="URL to the PDF document to be processed.")
    questions: List[str] = Field(..., description="A list of questions to be answered based on the document.")

class HackRxRunResponse(BaseModel):
    """Defines the structure for the /hackrx/run response body."""
    answers: List[str] = Field(..., description="A list of answers corresponding to the questions.")

# --- Global Variables and Initializations ---
# Initialize clients and models once to be reused across API calls.
# This is more efficient than re-initializing for every request.

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
PINECONE_INDEX_NAME = "hackrx-index"

# Initialize Google Gemini AI client
genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Initialize the sentence transformer model for creating embeddings.
# This model runs locally and converts text into numerical vectors.
# 'all-MiniLM-L6-v2' is a good balance of performance and speed.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIMENSION = 384 # Dimension of the 'all-MiniLM-L6-v2' model's embeddings

# --- Helper Functions ---

def download_and_read_pdf(url: str) -> str:
    """
    Downloads a PDF from a URL, extracts text from it, and returns the text.
    Handles potential download and processing errors.
    """
    try:
        logging.info(f"Downloading PDF from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Save the PDF to a temporary file
        temp_pdf_path = "temp_document.pdf"
        with open(temp_pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logging.info("PDF downloaded. Extracting text.")
        
        # Extract text from the downloaded PDF
        text = ""
        with open(temp_pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        
        os.remove(temp_pdf_path) # Clean up the temporary file
        logging.info(f"Extracted {len(text)} characters from the PDF.")
        return text
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Could not download PDF from URL: {e}")
    except Exception as e:
        logging.error(f"Failed to process PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read or process PDF: {e}")

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Splits a long text into smaller, overlapping chunks. This is crucial for
    creating meaningful embeddings without losing context at the boundaries.
    """
    if not text:
        return []
    
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
    """
    A context manager to safely handle the Pinecone index.
    It checks if the index exists, creates it if it doesn't, connects to it,
    and ensures it's cleaned up (deleted) after use. This is important for
    hackathons to manage resources and ensure each run is clean.
    """
    if PINECONE_INDEX_NAME in pc.list_indexes().names():
        logging.info(f"Deleting existing Pinecone index: {PINECONE_INDEX_NAME}")
        pc.delete_index(PINECONE_INDEX_NAME)

    logging.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine", # Cosine similarity is effective for semantic search
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT or "us-east-1") # Use serverless for cost-effectiveness
    )
    
    index = pc.Index(PINECONE_INDEX_NAME)
    try:
        yield index
    finally:
        logging.info(f"Deleting Pinecone index '{PINECONE_INDEX_NAME}' after processing.")
        pc.delete_index(PINECONE_INDEX_NAME)


def get_relevant_context(question: str, index) -> str:
    """
    Finds and returns the most relevant text chunks from the document for a given question.
    1. Creates an embedding for the question.
    2. Queries the Pinecone index to find the top N most similar text chunks.
    3. Joins these chunks into a single context string.
    """
    try:
        question_embedding = embedding_model.encode(question).tolist()
        query_result = index.query(
            vector=question_embedding,
            top_k=5, # Retrieve the top 5 most relevant chunks
            include_metadata=True
        )
        
        context_chunks = [match['metadata']['text'] for match in query_result['matches']]
        context = "\n---\n".join(context_chunks)
        logging.info(f"Retrieved context for question: '{question[:50]}...'")
        return context
    except Exception as e:
        logging.error(f"Error retrieving context from Pinecone: {e}")
        return "" # Return empty context on error

def answer_question_with_llm(question: str, context: str) -> str:
    """
    Uses the Gemini LLM to generate an answer to a question based on the provided context.
    The prompt is engineered to instruct the LLM to be concise and base its answer
    strictly on the given information.
    """
    if not context:
        return "I could not find relevant information in the document to answer this question."

    prompt = f"""
    Based *only* on the following context, please provide a clear and concise answer to the question.
    Do not use any information outside of this context. If the context does not contain the answer,
    say that the information is not available in the provided document.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    try:
        response = llm_model.generate_content(prompt)
        logging.info(f"Generated answer for question: '{question[:50]}...'")
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error generating answer with LLM: {e}")
        return "There was an error generating the answer."

# --- FastAPI Application Setup ---
app = FastAPI(
    title="HackRx 6.0 Intelligent Query-Retrieval System",
    description="API for processing documents and answering questions using RAG.",
    version="1.0.0"
)

# --- API Endpoints ---

@app.get("/", summary="Health Check", description="A simple endpoint to verify that the API is running.")
def read_root():
    """A simple health check endpoint to confirm the server is up."""
    return {"status": "ok", "message": "Welcome to the HackRx 6.0 API!"}


@app.post("/hackrx/run", 
          response_model=HackRxRunResponse,
          summary="Process Document and Answer Questions",
          description="The main endpoint for the hackathon. It takes a document URL and questions, and returns answers.")
async def hackrx_run(payload: HackRxRunRequest, _=Depends(get_current_user)):
    """
    The main logic for the hackathon submission.
    Orchestrates the entire RAG pipeline from document download to answer generation.
    """
    logging.info("Received request for /hackrx/run")
    
    # 1. Download and process the document
    document_text = download_and_read_pdf(payload.documents)
    text_chunks = chunk_text(document_text)
    
    if not text_chunks:
        raise HTTPException(status_code=500, detail="Failed to extract any text from the document.")

    # 2. Setup Pinecone index and upload document chunks
    with get_pinecone_index() as index:
        # --- OPTIMIZED SECTION ---
        logging.info("Generating embeddings for all chunks at once (this may take a moment)...")
        # Encode all chunks in a single, efficient batch call
        all_embeddings = embedding_model.encode(text_chunks, show_progress_bar=True)
        logging.info("Embeddings generated. Preparing for upsert.")

        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(text_chunks, all_embeddings)):
            vectors_to_upsert.append({
                "id": f"chunk_{i}",
                "values": embedding.tolist(),
                "metadata": {"text": chunk}
            })
        
        # Upsert in batches for efficiency
        logging.info(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
        index.upsert(vectors=vectors_to_upsert, batch_size=100)
        logging.info(f"Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone.")
        # --- END OF OPTIMIZED SECTION ---

        # 3. Process each question
        answers = []
        for question in payload.questions:
            # 3a. Retrieve relevant context
            context = get_relevant_context(question, index)
            # 3b. Generate answer based on context
            answer = answer_question_with_llm(question, context)
            answers.append(answer)

    logging.info("Successfully processed all questions. Returning response.")
    return HackRxRunResponse(answers=answers)

# --- To run this locally, use the command: ---
# uvicorn main:app --reload
