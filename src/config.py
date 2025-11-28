# src/config.py

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_storage")
    DATASET_PATH = os.getenv("DATASET_PATH", "datasets/golden_dataset.csv")
    DATA_DIR = os.getenv("DATA_DIR", "data")
    QUARANTINE_DIR = os.getenv("QUARANTINE_DIR", "data_quarantine")
    EMBEDDING_MODEL = "models/text-embedding-004" # Or "all-MiniLM-L6-v2"
    LLM_MODEL = "gemini-2.0-flash"
    LLM_PROVIDER ="GEMINI"  # or LOCAL
settings = Settings()