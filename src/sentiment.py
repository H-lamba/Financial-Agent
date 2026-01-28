from transformers import pipeline, AutoTokenizer
import torch
import re

## pipe = pipeline("text-classification", model = "ProsusAI/finbert") this loads model every time thisfile is called insead we can introduce lazzy loading pattern 
## Since finbert limit is of 512 tokens lets tokenize the text..... We will be using Sentence level chunking because it is fast and accurate it also preseves the semantic meaning of the words
_sentiment_pipeline = None
_tokenizer = None

def split_into_sentences(text):
    sentence = re.split(r'(?<=[.!?])\s+', text)
    return sentence

def chunk_by_sentences(text, tokenizer, max_tokens = 450):
    sentences = split_into_sentences(text)
    chunk = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens = False))
        if(current_length+sentence_tokens>max_tokens and current_chunk):
            chunk.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
        else :
            current_chunk.append(sentence)
            current_length += sentence_tokens
        
    if current_chunk:
        chunk.append(" ".join(current_chunk))
    return chunk    
            
            
            
def get_sentiment_pipeline():
    global _sentiment_pipeline
    global _tokenizer
    if _sentiment_pipeline is None:
        print("Model is loading ----- ")
        _sentiment_pipeline = pipeline(
            "text-classification",
            model = "ProsusAI/finbert",
            tokenizer = "ProsusAI/finbert",
            device = 0 if torch.cuda.is_available() else -1
        )
        _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        print("Model Loading is Successfull......")
        
    return _sentiment_pipeline, _tokenizer

def get_sentiment_by_sentence(text):
    if not text:
        return 0.0
    pipe, tokenizer = get_sentiment_pipeline()
    token_count = len(tokenizer.encode(text, add_special_tokens = False))
    if(token_count<= 512):
        result = pipe(text)[0]
        label = result["label"].lower()
        score = result["score"]
        if label == "positive":
            return score
        elif label == "negative":
            return -score
        else :
            return 0.0
    print(f"⚠️  Text has {token_count} tokens. Chunking by sentences...")
    chunks = chunk_by_sentences(text, tokenizer)
    print(f"   Split into {len(chunks)} chunks")
    
    # Analyze each chunk
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        result = pipe(chunk)[0]
        label = result["label"].lower()
        score = result["score"]
        
        if label == "positive":
            chunk_scores.append(score)
        elif label == "negative":
            chunk_scores.append(-score)
        else:
            chunk_scores.append(0.0)
        
        print(f"   Chunk {i+1}: {label} ({score:.3f})")
    
    avg_score = sum(chunk_scores) / len(chunk_scores)
    print(f"   Final score: {avg_score:.3f}")
    return avg_score

print("these are the results---------------")
print(get_sentiment_by_sentence('''The company announced today that it has exceeded all quarterly expectations 
    with record-breaking revenue of $5 billion. CEO John Smith stated that the 
    strong performance was driven by innovative product launches and expanding 
    market share in key regions. Analysts are praising the strategic decisions 
    made over the past year. However, some concerns remain about increasing 
    competition in the sector. The stock price surged 15% following the announcement.
    Despite positive earnings, the company warned of potential headwinds in the 
    coming quarter due to supply chain disruptions. Industry experts believe the 
    company is well-positioned to navigate these challenges given its strong 
    balance sheet and experienced management team.'''))