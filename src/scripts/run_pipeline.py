# src/scripts/run_pipeline.py
import sys
import os
import shutil
import logging
import asyncio

# 1. Add project root to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from dotenv import load_dotenv

# 2. Project Imports
from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import (
    VectorStoreService,
    run_indexing_service,
    run_retrieval_service
)
from src.infrastructure.llm.gemini_factory import GeminiFactory

# 3. Load Environment Variables
load_dotenv()

# ---------------- CONFIGURATION ----------------
DATA_DIR = "data"
QUARANTINE_DIR = "data_quarantine"
QDRANT_PATH = "qdrant_storage"
LOG_FILE = "rag_pipeline.log"

# ---------------- LOGGER SETUP ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ---------------- FACTORY FUNCTIONS ----------------
def get_rag_service() -> VectorStoreService:
    """
    Instantiates the RAG service with Gemini Embeddings and Qdrant.
    """
    # 1. Get the Gemini Embedder
    embedder = GeminiFactory.get_app_embedder(model_name="models/text-embedding-004")
    
    # 2. Connect to Qdrant
    qdrant_impl = QdrantImpl(collection_name="rag_chunks", path=QDRANT_PATH)
    
    return VectorStoreService(
        embedder=embedder,
        db_impl=qdrant_impl,
    )


# ---------------- PHASE 1: INDEXING ----------------
def run_indexing_phase(rag_service: VectorStoreService):
    """
    Iterates through DATA_DIR and indexes files. Moves failed files to QUARANTINE_DIR.
    """
    logger.info("🚀 Starting Indexing Phase")
    
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory '{DATA_DIR}' not found.")
        return

    os.makedirs(QUARANTINE_DIR, exist_ok=True)
    files_paths = os.listdir(DATA_DIR)

    if not files_paths:
        logger.warning("No files found to index.")
        return

    for file_name in files_paths:
        file_path = os.path.join(DATA_DIR, file_name)

        try:
            logger.info(f"Indexing file: {file_name}")
            
            run_indexing_service(
                file_path=file_path,
                vector_store=rag_service,
            )
            
            logger.info(f"✅ Successfully indexed: {file_name}")

        except Exception as e:
            logger.error(f"❌ Error processing {file_name}", exc_info=True)
            
            # Quarantine Logic
            quarantine_path = os.path.join(QUARANTINE_DIR, file_name)
            try:
                shutil.move(file_path, quarantine_path)
                logger.warning(f"⚠️ File moved to quarantine: {quarantine_path}")
            except Exception as move_error:
                logger.critical(
                    f"Failed to move {file_name} to quarantine: {move_error}",
                    exc_info=True
                )


# ---------------- PHASE 2: RETRIEVAL ----------------
def run_retrieval_test(rag_service: VectorStoreService, query_text: str):
    """
    Runs a sample query to verify the pipeline.
    """
    logger.info("🔎 Starting Retrieval Phase")
    logger.info(f"Query: {query_text}")

    results = run_retrieval_service(query_text, rag_service, top_k=3)

    if not results:
        logger.warning("No results found.")
        return

    for i, chunk in enumerate(results, start=1):
        logger.info(f"--- Result {i} ---")
        logger.info(f"Chunk ID:    {chunk.chunk_id}")
        logger.info(f"Source file: {chunk.source_file}")
        logger.info(f"Page:        {chunk.page}")
        logger.info(f"Type:        {chunk.type}")
        logger.debug(f"Content:     {chunk.content[:200]}...") # Log first 200 chars to avoid clutter


# ---------------- MAIN EXECUTION ----------------
async def main():
    logger.info("🔧 Initializing RAG Service...")
    rag_service = get_rag_service()

    # 1. Run Indexing (Comment out if you only want to query)
    run_indexing_phase(rag_service)

    # 2. Run Test Retrieval
    test_query = "what is the m/z of 4-Dihydroxyacetophenone"
    run_retrieval_test(rag_service, test_query)

    logger.info("🎉 Pipeline finished successfully")


if __name__ == "__main__":
    asyncio.run(main())