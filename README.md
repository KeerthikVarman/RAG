# PDF RAG System

This project is a complete End-to-End RAG (Retrieval-Augmented Generation) system for retrieving information from PDF files and generating answers using an LLM model.

The PDF files are loaded and split into smaller chunks. These chunks are converted into embeddings using the `all-MiniLM-L6-v2` model and stored in ChromaDB. When a user gives a query, the query is converted into an embedding to retrieve relevant content. The retrieved context chunks and the user's question are formatted into a structured RAG prompt and passed to an LLM model (`google/flan-t5-base`) to generate concise answers grounded in the document context.

## Technologies Used

* Python
* LangChain
* ChromaDB
* Sentence Transformers (`all-MiniLM-L6-v2`)
* PyMuPDF
* HuggingFace Transformers (`google/flan-t5-base`)

## How It Works

1. **Load PDF Files**: Extract text using `PyMuPDFLoader`
2. **Split Chunks**: Break text into chunks using `RecursiveCharacterTextSplitter`
3. **Generate Embeddings**: Convert chunks into 384-dimensional vectors using `all-MiniLM-L6-v2`
4. **Vector Store**: Upsert embeddings and document metadata into ChromaDB
5. **Retrieval**: Perform similarity query in ChromaDB for user question
6. **Prompt Formatting**: Structure context and user query into a grounded RAG prompt
7. **LLM Generation**: Feed prompt to LLM (`google/flan-t5-base`) to generate the final answer

## Models Used

* **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
* **LLM Model**: `google/flan-t5-base` (Text2Text Generation)

## Project Flow

```text
PDF Files
   ↓
Load Documents & Split into Chunks
   ↓
Generate Embeddings (SentenceTransformer)
   ↓
Store in Vector DB (ChromaDB)
   ↓
User Query
   ↓
Retrieve Relevant Chunks (RAGRetriever)
   ↓
Format RAG Prompt (RAGPrompt)
   ↓
Generate Answer (LLMModel)
   ↓
AI Response & Source Citations
```

## Installation

```bash
pip install -r requirement.txt
```

## Run

Place the PDF files inside the `data/pdf` folder and run:

```bash
python main.py
```

Or execute `python chat.py` directly for testing the pipeline.
