import os
import hashlib
import chromadb
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    print("Warning: GROQ_API_KEY is not set in .env file.")

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-20b",
    temperature=0.5,
    max_tokens=1024,
)


def load_pdfs(pdf_path="data/pdf"):
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


def rag_simple(query, retriever, llm, top_k=3):
    results = retriever.retrieve(query, top_k=top_k)

    context = ""
    for doc in results:
        context += (
            f"Source: {doc['metadata'].get('source_file')}\n"
            f"Page: {doc['metadata'].get('page')}\n"
            f"Content: {doc['document']}\n\n"
        )

    if not context:
        return "No relevant context found in documents."

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question only from the given context make with help of llm and give proper sentence.

If the answer is not available in the context , reply:
"."

Context:
{context}

Question:
{query}

Answer:
"""
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    vector_store = VectorStore()
    embedding_model = Embedding()
    
    if vector_store.collection.count() == 0:
        documents = load_pdfs("data/pdf")
        chunks = split_documents(documents)
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_model.generate_embedding(texts)
        vector_store.add_documents(chunks, embeddings)
    else:
        print("Vector database already exists. Skipping indexing.")

    retriever = RAGRetriever(
        vector_store,
        embedding_model
    )

    while True:
        try:
            query = input("\nAsk your question (type 'exit' to quit): ").strip()
            if not query:
                continue

            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            answer = rag_simple(
                query=query,
                retriever=retriever,
                llm=llm,
                top_k=3
            )

            print("\nAnswer:")
            print(answer)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break