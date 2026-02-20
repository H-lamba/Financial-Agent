FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (HF Spaces best practice)
RUN useradd -m -u 1000 user

# Set working directory and install deps as root first
WORKDIR /app

# Copy requirements first (better Docker cache)
COPY requirements.txt .

# Install CPU-only PyTorch first (smaller), then other deps
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY models/ ./models/

# Create directories for runtime
RUN mkdir -p documents data/faiss_index

# Change ownership to non-root user
RUN chown -R user:user /app

# Switch to non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose Streamlit port (HF Spaces expects port 7860)
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run Streamlit on port 7860 (Hugging Face Spaces default)
ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
