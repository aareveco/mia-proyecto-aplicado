
# MIA Proyecto Aplicado - Metabolomics RAG System

This project implements an advanced **Retrieval-Augmented Generation (RAG)** system specialized for **Metabolomics**, built using **Hexagonal Architecture**. 

It integrates **Hybrid Search** (combining Semantic and Keyword retrieval), **PubChem Data Augmentation**, and flexible **LLM Integration** (supporting both local Ollama models and Google Gemini) to provide accurate answers to scientific queries (e.g., Identifying M/Z values, retention times, and bioactivity).

## 🚀 Key Features

*   **Hexagonal Architecture**: Clean separation of Domain, Application Ports/Services, and Infrastructure Adapters for maintainability and scalability.
*   **Hybrid Search**: Combines **Dense Retrieval** (Qdrant) and **Keyword Search** (BM25) using Reciprocal Rank Fusion (RRF) for optimal context retrieval.
*   **PubChem Integration**: Automatically extracts chemical terms (like M/Z ratios) from queries and fetches real-time data from **PubChem** to augment the context.
*   **Unified LLM Factory**: Seamlessly switch between **Local LLMs** (via Ollama) and **Google Gemini** for query rewriting and answer generation.
*   **Intelligent Ingestion**: Automated PDF ingestion pipeline with metadata extraction and table processing.
*   **Evaluation**: Integrated **Ragas** evaluation pipeline for benchmarking retrieval and generation performance.

---

## 📂 Project Structure

The project follows a strict Hexagonal Architecture layout:

```text
.
├── .env                   # Environment variables (API Keys)
├── data/                  # Directory for input PDF documents
├── app.py                 # Streamlit UI Entry Point
├── pyproject.toml         # Dependencies managed by uv
└── src/
    ├── domain/            # Core Business Entities (ProcessedChunk, FilterSuggestion)
    ├── application/
    │   ├── ports/         # Interfaces (Ports) for external dependencies (LLMPort, IndexerPort)
    │   └── services/      # Application Business Logic (RAGService, IngestionService)
    └── infrastructure/
        ├── adapters/      # External integrations (PubChem, Qdrant, etc.)
        ├── llm/           # LLM Implementations (Gemini, Local)
        ├── retrieval/     # Search Strategies (BM25, RRF)
        └── evaluation/    # Ragas Evaluation Logic
```

---

## 🛠 Prerequisites

*   **Python 3.10+**
*   **[uv](https://github.com/astral-sh/uv)** (for fast, reliable dependency management)
*   **Ollama** (if using local models like `qwen2.5`)
*   **Google Cloud API Key** (if using Gemini)

---

## ⚡ Installation

1.  **Install `uv`** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd mia-proyecto-aplicado
    ```

3.  **Configure Environment**:
    Create a `.env` file in the root directory:
    ```ini
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```

4.  **Install Dependencies**:
    Sync the project environment:
    ```bash
    uv sync
    ```

---

## 🏃 How to Run

### 1. Start Local Services (Optional)
If you are strictly using **Gemini**, you can skip this. If you want to use local models:
```bash
ollama serve
# In a new terminal:
ollama pull qwen2.5:1.5b
```

### 2. Launch the Application
Start the Streamlit UI:
```bash
uv run streamlit run app.py
```

### 3. Using the App
1.  **Ingestion**: 
    *   Place your metabolomics PDF papers in the `data/` folder.
    *   In the sidebar, click **"Index Documents"**. The system will process, chunk, and index them into Qdrant and the BM25 store.
2.  **Search**:
    *   Type a query like: `what is the m/z of 4-Dihydroxyacetophenone?`
    *   The system will Rewrite the query -> Extract Filters -> Search PubChem -> Search Internal Docs -> Generate an Answer.
3.  **View Results**:
    *   See the generated answer, the retrieved chunks (with source highlighting), and any PubChem matches.

---

## 🧪 Evaluation & Datasets

To benchmark the RAG pipeline using Ragas:

**1. Generate Test Dataset:**
Create a golden dataset from your indexed documents:
```bash
uv run src/scripts/generate_test_dataset.py
```

**2. Run Evaluation:**
Execute the evaluation metrics (Context Precision, Recall, etc.):
```bash
uv run src/scripts/run_eval.py
```
*(Note: Ensure your local LLM or Gemini is configured correctly for the evaluation script)*