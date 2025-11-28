# src/scripts/run_pipeline.py
import sys
import os
import logging
import asyncio

# 1. Add project root to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from dotenv import load_dotenv

# 2. Project Imports
from src.infrastructure.services.service_factory import ServiceFactory
from src.application.services.indexing_service import BatchIndexingService
from src.application.services.rag_service import run_retrieval_service

# 3. Load Environment Variables
load_dotenv()

# ---------------- CONFIGURATION ----------------
DATA_DIR = "data"
QUARANTINE_DIR = "data_quarantine"
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


# ---------------- PHASE 2: RETRIEVAL ----------------
def run_retrieval_test(rag_service, query_text: str):
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
    rag_service = ServiceFactory.get_vector_service()

    # 1. Run Indexing (Comment out if you only want to query)
    logger.info("🚀 Starting Indexing Phase")
    indexing_service = BatchIndexingService(rag_service, quarantine_dir=QUARANTINE_DIR)
    indexing_service.index_directory(DATA_DIR)

    # 2. Run Test Retrieval
    test_query = "what is the m/z of 4-Dihydroxyacetophenone"
    run_retrieval_test(rag_service, test_query)

    logger.info("🎉 Pipeline finished successfully")


if __name__ == "__main__":
    asyncio.run(main())
