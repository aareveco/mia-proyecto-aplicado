
---

RAG Hito 1: Clean Architecture

This project implements a Retrieval-Augmented Generation (RAG) system using Hexagonal Architecture.  
It uses Streamlit for the UI and Qdrant for vector storage.

Quick Start

This project uses uv for fast dependency management.

1. Prerequisites

Ensure you have uv installed:

MacOS / Linux  
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh 
```

Windows  
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Project Structure
```
.
├── .env                   Environment variables (API Keys)  
├── data/                  Place your PDF documents here  
├── app.py                 Streamlit Entry Point (UI)  
├── main.py                CLI Entry Point  
├── pyproject.toml         Dependencies (optional if using uv directly)  
└── src/  
    ├── domain/            Entities (ProcessedChunk, etc.)  
    ├── application/       Ports (Interfaces) & Services (Use Cases)  
    └── infrastructure/    Adapters (PDFLoader, Qdrant, HuggingFace)
```

To install depdencies for local run:

```bash
brew install ollama
```

```bash
ollama serve
ollama pull qwen2.5:1.5b
```


To run use:
```bash
uv sync  
uv run streamlit run app.py
```

To Run generate test dataset or run eval use :

```bash
uv run src/scripts/generate_test_dataset.py
uv run python src/scripts/run_eval.py
```





Example query:
```
what is the m/z of 4-Dihydroxyacetophenone ?
```
---

