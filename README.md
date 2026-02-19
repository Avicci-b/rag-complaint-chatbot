# RAG Complaint Chatbot for CrediTrust Financial

An intelligent complaint analysis system that uses Retrieval-Augmented Generation (RAG) to answer questions about customer complaints from the CFPB dataset.

## Project Structure

- `data/`: raw and processed datasets
- `vector_store/`: persisted FAISS/ChromaDB indices
- `notebooks/`: Jupyter notebooks for EDA and experimentation
- `src/`: Python modules for the RAG pipeline
- `tests/`: unit tests
- `app.py`: Gradio/Streamlit web interface

## Setup

1. Create virtual environment: `python3 -m venv venv`
2. Activate: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
