import os
import hashlib
import chromadb
import numpy as np
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def load_pdfs(pdf_path):
    all_documents = []
    pdf_files = list(Path(pdf_path).glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        try:
            print(f"Loading {pdf_file.name}")
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")

    print(f"Total pages loaded: {len(all_documents)}")
    return all_documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")
    return chunks


class Embedding:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def generate_embedding(self, texts):
        print(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Embedding shape: {embeddings.shape}")
        return embeddings


class VectorStore:
    def __init__(self, collection_name="pdf_documents", persist_directory="data/vector_store"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        print(f"Existing chunks: {self.collection.count()}")

    def add_documents(self, documents, embeddings):
        ids = []
        texts = []
        metadatas = []
        embedding_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", 0)
            doc_id = hashlib.md5(f"{source}_{page}_{i}".encode()).hexdigest()

            ids.append(doc_id)
            texts.append(doc.page_content)
            metadatas.append(dict(doc.metadata))
            embedding_list.append(embedding.tolist())

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embedding_list
        )

        print(f"Stored {len(documents)} chunks")
        print(f"Total chunks in database: {self.collection.count()}")


class RAGRetriever:
    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query, top_k=5):
        print(f"User query: {query}")
        query_embedding = self.embedding_manager.generate_embedding([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                retrieved_docs.append({
                    "id": doc_id,
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                    "rank": i + 1
                })

            print(f"Retrieved {len(retrieved_docs)} chunks")
            return retrieved_docs

        except Exception as e:
            print(f"Retrieval error: {e}")
            return []


documents = load_pdfs("data/pdf")
chunks = split_documents(documents)

embedding_model = Embedding()

texts = [doc.page_content for doc in chunks]
embeddings = embedding_model.generate_embedding(texts)

vector_store = VectorStore()
vector_store.add_documents(chunks, embeddings)

retriever = RAGRetriever(vector_store, embedding_model)

results = retriever.retrieve("What is the source file name?")

for result in results:
    print("Rank:", result["rank"])
    print("Source:", result["metadata"].get("source_file"))
    print("Page:", result["metadata"].get("page"))
    print("Distance:", result["distance"])
    print("Content:", result["document"][:300])
    print("-" * 50)