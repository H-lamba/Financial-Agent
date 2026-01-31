import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List

try:
    from src.rag.embedding import get_embedding_model
except ImportError:
    from embedding import get_embedding_model

DB_PATH = "data/faiss_index"


def create_vector_db(chunks: List[Document]):
    embedding_model = get_embedding_model()
    print(f"Creating vector store for {len(chunks)} chunks...")
    
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(DB_PATH)
    
    print(f"✅ Vector store created and saved to {DB_PATH}")
    return db


def load_vector_db():
    embedding_model = get_embedding_model()
    
    if not os.path.exists(DB_PATH):
        print("❌ Vector store not found")
        return None

    print(f"📂 Loading vector store from {DB_PATH}")
    
    db = FAISS.load_local(
        DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

if __name__ == "__main__":
    try:
        try:
            from loader import load_document
            from splitter import split_documents
        except ImportError:
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from src.rag.loader import load_document
            from src.rag.splitter import split_documents

        print("Phase 1 Processing Started")
        docs = load_document("documents/ITC-Report-and-Accounts-2025.pdf")
        print(f"✅ Phase 1 Processing Completed")
        chunks = split_documents(docs)

        print("Vectorizing ")
        db = create_vector_db(chunks)

        print("Search Test")
        query = "What is the revenue growth?"
        results = db.similarity_search(query, k=2)
        print("\nSearch Results:")

        print(f"Query : {query}")
        print(f"Results : {results}")
        print(f"Top match : {results[0].page_content[:200]}...")

    except Exception as e:
        print(f"❌ Error during test: {e}")       