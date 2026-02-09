import os
import sys

# --- ROBUST IMPORT LOGIC ---
try:
    # 1. Try Absolute Import (Standard for 'main.py' or 'agent.py' from root)
    from src.rag.loader import load_document
    from src.rag.splitter import split_documents
    from src.rag.vectorstore import create_vector_db
    from src.rag.rag_chain import RAGChain
except ImportError:
    try:
        # 2. Try Package Import (For Streamlit/Agent when 'src' is in path)
        from rag.loader import load_document
        from rag.splitter import split_documents
        from rag.vectorstore import create_vector_db
        from rag.rag_chain import RAGChain
    except ImportError:
        # 3. Fallback (Running script directly inside 'rag' folder)
        from loader import load_document
        from splitter import split_documents
        from vectorstore import create_vector_db
        from rag_chain import RAGChain

class DocumentTool:
    def __init__(self):
        self.chain = None
        # Check for existing vector store
        # We check multiple relative paths to find the data
        possible_paths = ["data/faiss_index", "../data/faiss_index", "../../data/faiss_index"]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"   (Found knowledge base at: {path})")
                self.chain = RAGChain()
                break

    def ingest(self, pdf_path):
        print(f"⚙️  Ingesting document: {pdf_path}")
        
        docs = load_document(pdf_path)
        if not docs:
            return False
        
        chunks = split_documents(docs)
        if not chunks:
            return False
        
        create_vector_db(chunks)
        
        # Reload chain with new data
        self.chain = RAGChain()
        return True

    def query(self, question):
        if not self.chain:
            return {
                "answer": "⚠️ No document loaded. Please upload a PDF first.",
                "source_text": ""
            }
        
        response = self.chain.answer_query(question)
        
        answer = response['answer']
        source = ""
        if response.get('source_documents'):
            source = response['source_documents'][0].page_content[:200]
        
        return {
            "answer": answer,
            "source_text": source
        }