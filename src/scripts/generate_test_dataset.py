
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.infrastructure.loaders.pdf_loader import PDFLoader
from src.application.services.dataset_generation_service import DatasetGenerationService

def main():
    # --- Configuration ---
    DATA_DIR = "data"          # Directory containing multiple PDFs
    OUTPUT_DIR = "datasets"    # Output folder
    TEST_SIZE = 10             # How many Q&A pairs to generate total
    # ---------------------
    
    # 1. Instantiate Loader
    loader = PDFLoader()
    
    # 2. Instantiate Service
    service = DatasetGenerationService(loader=loader)
    
    # 3. Execute
    if os.path.isdir(DATA_DIR):
        service.run(input_dir=DATA_DIR, output_dir=OUTPUT_DIR, test_size=TEST_SIZE)
    else:
        print(f"Error: Directory '{DATA_DIR}' not found.")

if __name__ == "__main__":
    main()
