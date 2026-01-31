# src/rag/splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    
    # Initialize the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    # ✅ FIX: Use 'text_splitter' variable defined above
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    # Test Block
    try:
        from loader import load_document
        
        # Test with your specific file
        test_file = "documents/ITC-Report-and-Accounts-2025.pdf"
        print(f"📄 Loading {test_file}...")
        
        docs = load_document(test_file)
        chunks = split_documents(docs)

        print(f"✅ Success!")
        print(f"Original Pages: {len(docs)}")
        print(f"Total Chunks:   {len(chunks)}")

        if chunks:
            print("\n🔍 Sample Chunk Content:")
            print("-" * 40)
            print(chunks[0].page_content[:500])
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Error during test: {e}")