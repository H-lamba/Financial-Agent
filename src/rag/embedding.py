from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    print("Loading the model for the embedding from the hugging face")
    model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    return model
if __name__ == "__main__":
    # Test Block
    try:
        model = get_embedding_model()
        vector = model.embed_query("Testing financial embeddings")
        print(f"✅ Model Loaded Successfully!")
        print(f"📊 Vector Dimension: {len(vector)} (Should be 384)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")