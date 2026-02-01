import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env
load_dotenv()

# Import handling
try:
    from src.rag.vectorstore import load_vector_db
except ImportError:
    from vectorstore import load_vector_db

class RAGChain:
    def __init__(self):
        print("🔄 Connecting to gemini-3-flash-preview (Cloud)...")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0,
            convert_system_message_to_human=True
        )
        self.db = load_vector_db()
        if self.db is None:
            print("⚠️ Warning: Vector Store not found. Please run vectorstore.py first.")
        else:
            print("✅ Connected to Knowledge Base.")
    def get_chain(self):
        if self.db is None:
            return None
        template = """You are a financial analyst. Answer the question strictly based on the provided context.
        If the answer is not in the context, say "Data not found in the document."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        return qa_chain
    
    def answer_query(self, query):
        chain = self.get_chain()
        if not chain:
            return {"answer": "Vector Store not loaded.", "source_documents": []}
            
        result = chain.invoke({"query": query})
        
        return {
            "answer": result['result'],
            "source_documents": result['source_documents']
        }

if __name__ == "__main__":
    # Test Block
    try:
        rag = RAGChain()
        if rag.db:
            test_query = "What is the revenue growth and future outlook?"
            print(f"\n❓ Question: {test_query}")
            print("⏳ Asking Gemini...")
            
            response = rag.answer_query(test_query)
            
            print(f"\n💡 Gemini Answer:\n{response['answer']}")
            
            if response['source_documents']:
                print(f"\n📄 Source: {response['source_documents'][0].page_content[:150]}...")
    except Exception as e:
        print(f"❌ Error: {e}")