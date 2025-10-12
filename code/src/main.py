"""
main.py — Gemini RAG Pipeline
----------------------------------
End-to-end process for:
1. Preparing text chunks
2. Generating or loading embeddings
3. Storing them in Chroma
4. Querying Gemini LLM with retrieval context

Includes:
- Exception handling and validation
- Logging for debugging
- Deterministic behavior (seeded randomness)
"""

import os
import pickle
import logging
import random
import google.generativeai as genai
from preprocess import prepare_chunks
from vector_store import ChromaRAG
from prompts import make_rag_prompt
from embedder import GeminiEmbedder
from config import Config
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# Setup
# -------------------------------
random.seed(getattr(Config, "SEED", 42))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------
# 1. Load or Create Chunks
# -------------------------------
try:
    if not hasattr(Config, "DATA_PATH") or not os.path.exists(Config.DATA_PATH):
        raise FileNotFoundError("❌ DATA_PATH not found in Config or file missing.")
    logger.info("🔄 Preparing text chunks from dataset...")
    all_chunks, metadata_list = prepare_chunks(input_file=Config.DATA_PATH)
    if not all_chunks:
        raise ValueError("❌ No chunks generated from input file.")
    logger.info(f"✅ Prepared {len(all_chunks)} chunks successfully.")
except Exception as e:
    logger.exception(f"Failed to prepare chunks: {e}")
    raise


# -------------------------------
# 2. Create or Load Embeddings
# -------------------------------
try:
    if not os.path.exists(Config.EMBEDDINGS_FILE):
        logger.info("📦 No embeddings.pkl found — generating new embeddings...")
        embedder = GeminiEmbedder()
        embeddings = embedder.embed_documents(all_chunks)

        with open(Config.EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(embeddings, f)
        logger.info(f"✅ Created {len(embeddings)} embeddings and saved to {Config.EMBEDDINGS_FILE}.")
    else:
        logger.info("✅ embeddings.pkl already exists. Skipping embedding generation.")
        with open(Config.EMBEDDINGS_FILE, "rb") as f:
            embeddings = pickle.load(f)
except Exception as e:
    logger.exception(f"Failed to generate or load embeddings: {e}")
    raise


# -------------------------------
# 3. Initialize Chroma RAG Store
# -------------------------------
try:
    rag_store = ChromaRAG(db_path=Config.CHROMA_DB, collection_name=Config.collection_name)
    logger.info("✅ Chroma RAG store initialized successfully.")
except Exception as e:
    logger.exception(f"Failed to initialize ChromaRAG: {e}")
    raise


# -------------------------------
# 4. Query Gemini + Retrieval
# -------------------------------
def query_gemini(prompt: str) -> str:
    """Send the constructed prompt to Gemini and return the text output."""
    try:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        genai.configure(api_key=os.getenv("GEMINI_FREE_API_KEY"))
        model = genai.GenerativeModel(Config.GEMINI_LLM_MODEL)
        answer = model.generate_content(prompt)
        logger.info("✅ Gemini model response received.")
        return answer.text
    except Exception as e:
        logger.exception(f"Gemini query failed: {e}")
        return f"Error generating response: {e}"


def generate_response(query_input: str) -> str:
    """Retrieve context from RAG store and query Gemini."""
    try:
        if not query_input.strip():
            raise ValueError("Query input cannot be empty.")
        response, all_docs = rag_store.retrieve(query_input, top_k=3)
        logger.info(f"🔍 Retrieved {len(response)} relevant passages for the query.")
        rag_prompt = make_rag_prompt(
            query=query_input,
            relevant_passage="\n".join(response)
        )
        return query_gemini(rag_prompt)
    except Exception as e:
        logger.exception(f"Error generating response: {e}")
        return f"Error generating response: {e}"


# -------------------------------
# 5. Run Main
# -------------------------------
if __name__ == "__main__":
    try:
        query = (
            "I'm looking to purchase the yearly premium plan for $17.00, "
            "which comes with a free domain for one year. However, the voucher "
            "isn't appearing at checkout. Should I complete the purchase first "
            "for the voucher to become available for activation?"
        )
        logger.info("🚀 Retrieval starts...")
        answer = generate_response(query)
        logger.info("Response:\n" + answer)
        print("Response:", answer)
    except Exception as e:
        logger.exception(f"❌ Fatal error in main execution: {e}")
