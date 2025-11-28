import os
import shutil
import logging
from typing import List, Optional

from src.application.services.rag_service import VectorStoreService, run_indexing_service

logger = logging.getLogger(__name__)

class BatchIndexingService:
    """
    Service responsible for batch indexing files from a directory.
    Handles iterating over files, calling the indexing service, and error handling (quarantine).
    """

    def __init__(self, rag_service: VectorStoreService, quarantine_dir: Optional[str] = None):
        self.rag_service = rag_service
        self.quarantine_dir = quarantine_dir

    def index_directory(self, data_dir: str, on_progress: Optional[callable] = None) -> None:
        """
        Indexes all PDF files in the given directory.

        Args:
            data_dir: Directory containing files to index.
            on_progress: Optional callback function(index, total, current_file) for progress updates.
        """
        if not os.path.exists(data_dir):
            logger.error(f"Data directory '{data_dir}' not found.")
            return

        files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")] # Filtering for PDFs as per usage

        if not files:
            logger.warning("No files found to index.")
            return

        if self.quarantine_dir:
             os.makedirs(self.quarantine_dir, exist_ok=True)

        total_files = len(files)

        for i, file_name in enumerate(files):
            file_path = os.path.join(data_dir, file_name)

            if on_progress:
                on_progress(i, total_files, file_name)

            try:
                logger.info(f"Indexing file: {file_name}")
                run_indexing_service(
                    file_path=file_path,
                    vector_store=self.rag_service,
                )
                logger.info(f"✅ Successfully indexed: {file_name}")

            except Exception as e:
                logger.error(f"❌ Error processing {file_name}", exc_info=True)

                if self.quarantine_dir:
                    quarantine_path = os.path.join(self.quarantine_dir, file_name)
                    try:
                        shutil.move(file_path, quarantine_path)
                        logger.warning(f"⚠️ File moved to quarantine: {quarantine_path}")
                    except Exception as move_error:
                        logger.critical(
                            f"Failed to move {file_name} to quarantine: {move_error}",
                            exc_info=True
                        )
                else:
                    logger.warning(f"Skipping {file_name} due to error (no quarantine configured).")

        if on_progress:
            on_progress(total_files, total_files, "Done")
