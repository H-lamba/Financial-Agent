import os
import sys

# Dynamic imports to handle running as a module vs script
try:
    from src.rag.loader import load_document
    from src.rag.splitter import split_documents
    from src.rag.vectorstore import create_vector_db
    from src.rag.rag_chain import RAGChain
except ImportError:
    # Fallback for local testing
    from loader import load_document
    from splitter import split_documents
    from vectorstore import create_vector_db
    from rag_chain import RAGChain

class DocumentTool:
    def __init__(self):
        self.chain = None
        # Check if we already have a vector store (memory)
        if os.path.exists("data/faiss_index"):
            self.chain = RAGChain()

    def ingest(self, pdf_path):
        """
        Takes a PDF, reads it, chops it up, and saves it to the vector database.
        """
        print(f"⚙️  Ingesting document: {pdf_path}")
        
        # 1. Load PDF
        docs = load_document(pdf_path)
        if not docs:
            print("❌ Failed to load document.")
            return False
        
        # 2. Split into chunks
        chunks = split_documents(docs)
        if not chunks:
            print("❌ Failed to split document.")
            return False
        
        # 3. Create Vector Store (Saves to 'data/faiss_index')
        create_vector_db(chunks)
        
        # 4. Initialize the Chain (Now that data exists)
        self.chain = RAGChain()
        return True

    def query(self, question):
        """
        Asks the RAG Chain a question.
        """
        if not self.chain:
            return {
                "answer": "⚠️ No document loaded. Please load a PDF first.",
                "source_text": ""
            }
        
        # Run the query through Gemini/RAG
        response = self.chain.answer_query(question)
        
        # Format the output for the Agent
        answer = response['answer']
        source = ""
        
        # Safely extract source text if available
        if response.get('source_documents'):
            source = response['source_documents'][0].page_content[:200]
        
        return {
            "answer": answer,
            "source_text": source
        }