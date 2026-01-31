from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pathlib import Path
from typing import List
from langchain_core.documents import Document
import os
def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    documents =  loader.load()
    return documents

def load_text(file_path: str) -> List[Document]:
    loader = TextLoader(file_path, encoding = "utf-8")
    documents = loader.load()
    return documents

def load_document(file_path : str) -> List[Document]:
    path = Path(file_path)
    if not path.exists():
        raise ValueError("File does not exist")
    suffix = path.suffix.lower() 
    if suffix == ".pdf":
        return load_pdf(file_path)
    elif suffix == ".txt":
        return load_text(file_path)
    else:
        raise ValueError("Unsupported file format")
if __name__ == "__main__":
    # Test file path
    test_path = "documents/ITC-Report-and-Accounts-2025.pdf"
    
    try:
        # Check if directory exists, if not create it (optional helper)
        if not os.path.exists("documents"):
            os.makedirs("documents")
            print("⚠️ Created 'documents' folder. Please put a PDF there to test.")
        
        docs = load_document(test_path)
        print(f"✅ Successfully loaded {len(docs)} pages.")
        print(f"📄 Sample Content (Page 1):\n{docs[0].page_content[:500]}...")
        
    except ValueError as e:
        print(f"⚠️ Test Skipped: {e}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")