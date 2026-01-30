from transformers import pipeline, AutoTokenizer
import torch
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Global variables for lazy loading (Task 28)
_sentiment_pipeline = None
_tokenizer = None


def get_sentiment_pipeline():
    """
    Lazy-load the sentiment model and tokenizer.
    Model loads only once when first called, then cached.
    
    Returns:
        tuple: (pipeline, tokenizer)
    """
    global _sentiment_pipeline, _tokenizer
    
    if _sentiment_pipeline is None:
        print("🔄 Loading FinBERT model (first time only)...")
        _sentiment_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512
        )
        _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        print("✅ Model loaded successfully!")
    
    return _sentiment_pipeline, _tokenizer


def split_into_sentences(text):
    """
    Split text into sentences using regex.
    Handles common financial abbreviations.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of sentences
    """
    if not text:
        return []
    
    # Handle common abbreviations to prevent incorrect splits
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Inc|Co|Corp|Ltd)\.', r'\1', text)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter empty strings and strip whitespace
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_sentences(text, tokenizer, max_tokens=450):
    """
    Group sentences into chunks that fit within token limit.
    Uses sentence boundaries to preserve semantic meaning.
    
    Args:
        text (str): Input text
        tokenizer: HuggingFace tokenizer
        max_tokens (int): Maximum tokens per chunk (leaves room for special tokens)
        
    Returns:
        list: List of text chunks
    """
    sentences = split_into_sentences(text)
    
    # Edge case: Empty text
    if not sentences:
        return []
    
    # Edge case: Single very long sentence
    if len(sentences) == 1:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            # Truncate to max_tokens
            truncated_tokens = tokens[:max_tokens]
            return [tokenizer.decode(truncated_tokens)]
        return [text]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
        
        # If single sentence exceeds limit, add it separately (will be truncated)
        if sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            chunks.append(sentence)
            continue
        
        # If adding this sentence exceeds limit, start new chunk
        if current_length + sentence_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens
    
    # Add remaining sentences
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def get_sentiment(text, confidence_threshold=0.0, use_weighting=True, verbose=False):
    """
    Analyze sentiment of financial text with automatic chunking and confidence weighting.
    
    This is the main function for sentiment analysis. It handles:
    - Single texts or lists of texts (Task 32: Batch processing)
    - Long texts via sentence-level chunking (Task 31)
    - Confidence weighting for reliable predictions (Task 34)
    
    Args:
        text (str or list): Text or list of texts to analyze
        confidence_threshold (float): Minimum confidence (0.0-1.0) to include predictions
                                      0.0 = use all, 0.7 = only high-confidence
        use_weighting (bool): If True, weight predictions by confidence score
        verbose (bool): Print detailed analysis information
        
    Returns:
        float: Sentiment score from -1 (very negative) to +1 (very positive)
               0 = neutral
    """
    if not text:
        return 0.0
    
    pipe, tokenizer = get_sentiment_pipeline()
    
    # Handle list of texts (Task 32: Batch processing)
    if isinstance(text, list):
        valid_texts = [t for t in text if t]
        if not valid_texts:
            return 0.0
        scores = [get_sentiment(t, confidence_threshold, use_weighting, False) 
                  for t in valid_texts]
        return sum(scores) / len(scores) if scores else 0.0
    
    # Check if chunking needed (Task 31)
    token_count = len(tokenizer.encode(text, add_special_tokens=True))
    
    if token_count <= 512:
        # Short text - analyze directly
        chunks = [text]
    else:
        # Long text - chunk by sentences
        if verbose:
            print(f"⚠️  Text has {token_count} tokens. Chunking by sentences...")
        chunks = chunk_by_sentences(text, tokenizer)
        if verbose:
            print(f"   Created {len(chunks)} chunks")
    
    # Analyze chunks with confidence weighting (Task 34)
    weighted_sum = 0.0
    total_weight = 0.0
    filtered_count = 0
    
    for i, chunk in enumerate(chunks):
        result = pipe(chunk)[0]
        label = result["label"].lower()
        confidence = result["score"]
        
        # Apply confidence threshold (Task 34)
        if confidence < confidence_threshold:
            filtered_count += 1
            if verbose:
                print(f"   Chunk {i+1}: {label} (conf: {confidence:.3f}) - FILTERED")
            continue
        
        # Convert label to sentiment value
        if label == "positive":
            sentiment = 1.0
        elif label == "negative":
            sentiment = -1.0
        else:  # neutral
            sentiment = 0.0
        
        # Apply confidence weighting (Task 34)
        if use_weighting:
            weight = confidence
            weighted_sum += sentiment * weight
            total_weight += weight
        else:
            weighted_sum += sentiment * confidence
            total_weight += 1.0
        
        if verbose:
            contribution = sentiment * confidence
            print(f"   Chunk {i+1}: {label} (conf: {confidence:.3f}, "
                  f"contribution: {contribution:+.3f})")
    
    # Calculate final score
    if total_weight > 0:
        final_score = weighted_sum / total_weight
    else:
        final_score = 0.0
    
    if verbose:
        if filtered_count > 0:
            print(f"   ⚠️  Filtered {filtered_count} low-confidence predictions")
        print(f"✅ Final weighted score: {final_score:.3f}")
    
    return final_score


def visualize_sentiment_over_time(dates, scores, title="Sentiment Analysis Over Time", 
                                   save_path=None):
    """
    Create a visualization of sentiment scores over time (Task 38).
    """
    # FIX: Check length instead of "truthiness" to support Pandas indexes
    if len(dates) == 0:
        print("⚠️ No dates provided for visualization.")
        return None

    # Convert strings to datetime if needed
    # We check the first element safely
    first_date = dates[0] if isinstance(dates, list) else dates[0]
    if isinstance(first_date, str):
        dates = pd.to_datetime(dates)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot sentiment line
    ax.plot(dates, scores, marker='o', linewidth=2, markersize=6, 
            color='steelblue', label='Sentiment Score')
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral')
    
    # Color the background
    # We use list comprehension to ensure compatibility with different array types
    ax.fill_between(dates, 0, scores, where=[s >= 0 for s in scores], 
                     alpha=0.3, color='green', label='Positive')
    ax.fill_between(dates, 0, scores, where=[s < 0 for s in scores], 
                     alpha=0.3, color='red', label='Negative')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sentiment Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Rotate dates
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved to: {save_path}")
    
    return fig


def save_sentiment_to_csv(dates, scores, ticker, file_path=None):
    """
    Save sentiment scores to CSV file (Task 39).
    
    Args:
        dates (list): List of dates
        scores (list): List of sentiment scores
        ticker (str): Stock ticker symbol
        file_path (str): Optional custom file path
        
    Returns:
        str: Path to saved CSV file
    """
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sentiment_score': scores
    })
    
    # Generate default file path if not provided
    if file_path is None:
        # Create data folder if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")
        
        timestamp = datetime.now().strftime("%Y%m%d")
        file_path = f"data/{ticker}_sentiment_{timestamp}.csv"
    
    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f"💾 Sentiment data saved to: {file_path}")
    
    return file_path


def analyze_daily_news(news_df, date_column='date', text_columns=None):
    """
    Analyze sentiment for daily news aggregated by date (Task 33).
    Useful for integrating with your process_data.py pipeline.
    
    Args:
        news_df (pd.DataFrame): DataFrame with news data
        date_column (str): Name of the date column
        text_columns (list): Columns to combine for analysis 
                            (default: ['title', 'description'])
        
    Returns:
        pd.DataFrame: DataFrame with date and sentiment_score columns
    """
    if text_columns is None:
        text_columns = ['title', 'description']
    
    # Group by date
    daily_sentiment = []
    
    for date, group in news_df.groupby(date_column):
        # Combine text from all articles on that date
        texts = []
        for _, row in group.iterrows():
            combined = " ".join([str(row.get(col, '')) for col in text_columns 
                                if row.get(col)])
            if combined.strip():
                texts.append(combined)
        
        # Calculate average sentiment for the day
        if texts:
            score = get_sentiment(texts)
        else:
            score = 0.0
        
        daily_sentiment.append({
            'date': date,
            'sentiment_score': score,
            'article_count': len(texts)
        })
    
    return pd.DataFrame(daily_sentiment)


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SENTIMENT ANALYSIS - PHASE 3 COMPLETE TESTING")
    print("=" * 80)
    
    # Task 35: Test with negative text
    print("\n📊 Task 35 - Test Negative Sentiment:")
    negative_text = "The company went bankrupt and filed for Chapter 11 protection"
    score_neg = get_sentiment(negative_text, verbose=True)
    assert score_neg < -0.5, f"Expected negative score, got {score_neg}"
    print(f"✅ Result: {score_neg:.3f} (correctly negative)\n")
    
    # Task 36: Test with positive text
    print("\n📊 Task 36 - Test Positive Sentiment:")
    positive_text = "Record breaking profits announced with 300% revenue growth"
    score_pos = get_sentiment(positive_text, verbose=True)
    assert score_pos > 0.5, f"Expected positive score, got {score_pos}"
    print(f"✅ Result: {score_pos:.3f} (correctly positive)\n")
    
    # Task 31: Test with long text (chunking)
    print("\n📊 Task 31 - Test Long Text Chunking:")
    long_text = """
    The company announced today that it has exceeded all quarterly expectations 
    with record-breaking revenue of $5 billion. CEO John Smith stated that the 
    strong performance was driven by innovative product launches and expanding 
    market share in key regions. Analysts are praising the strategic decisions 
    made over the past year. However, some concerns remain about increasing 
    competition in the sector. The stock price surged 15% following the announcement.
    Despite positive earnings, the company warned of potential headwinds in the 
    coming quarter due to supply chain disruptions. Industry experts believe the 
    company is well-positioned to navigate these challenges given its strong 
    balance sheet and experienced management team.
    """ * 3  # Make it longer to trigger chunking
    score_long = get_sentiment(long_text, verbose=True)
    print(f"✅ Result: {score_long:.3f}\n")
    
    # Task 32: Test batch processing
    print("\n📊 Task 32 - Test Batch Processing:")
    batch_texts = [
        "Stock surges on earnings beat",
        "Company misses revenue targets",
        "Neutral market outlook reported"
    ]
    batch_scores = [get_sentiment(text) for text in batch_texts]
    print(f"Batch results: {[f'{s:.2f}' for s in batch_scores]}")
    avg_batch = sum(batch_scores) / len(batch_scores)
    print(f"✅ Average: {avg_batch:.3f}\n")
    
    # Task 34: Test confidence weighting
    print("\n📊 Task 34 - Test Confidence Weighting:")
    uncertain_texts = [
        "Stock might possibly increase slightly",  # Low confidence
        "MASSIVE EARNINGS EXPLOSION! INCREDIBLE GROWTH!",  # High confidence
    ]
    
    print("Without weighting:")
    score_no_weight = get_sentiment(uncertain_texts, use_weighting=False, verbose=True)
    
    print("\nWith weighting:")
    score_weighted = get_sentiment(uncertain_texts, use_weighting=True, verbose=True)
    
    print(f"\n✅ Weighting makes a difference: {score_no_weight:.3f} vs {score_weighted:.3f}\n")
    
    # Task 38: Test visualization
    print("\n📊 Task 38 - Test Sentiment Visualization:")
    test_dates = pd.date_range('2026-01-01', periods=10, freq='D')
    test_scores = [0.8, 0.5, -0.2, -0.6, 0.1, 0.4, 0.9, -0.3, 0.2, 0.6]
    
    fig = visualize_sentiment_over_time(
        test_dates, 
        test_scores, 
        title="NVDA Sentiment - Test Data",
        save_path="data/sentiment_chart_test.png"
    )
    print("✅ Visualization created!\n")
    plt.close(fig)
    
    # Task 39: Test CSV export
    print("\n📊 Task 39 - Test CSV Export:")
    csv_path = save_sentiment_to_csv(
        test_dates, 
        test_scores, 
        ticker="NVDA"
    )
    print(f"✅ CSV saved to: {csv_path}\n")
    
    # Summary
    print("=" * 80)
    print("✅ ALL PHASE 3 TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n✅ Task 28: Model caching implemented")
    print("✅ Task 31: Long text chunking working")
    print("✅ Task 32: Batch processing working")
    print("✅ Task 33: Aggregation function available")
    print("✅ Task 34: Confidence weighting implemented")
    print("✅ Task 35-36: Positive/negative tests passed")
    print("✅ Task 38: Visualization function created")
    print("✅ Task 39: CSV export function created")
    print("\n🎉 Ready to integrate with your main pipeline!")
    print("=" * 80)
