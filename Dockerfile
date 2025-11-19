# Enhanced SentimentR
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Copy application code
COPY enhanced_sentimentr/ ./enhanced_sentimentr/
COPY setup.py pyproject.toml ./

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 sentimentr && chown -R sentimentr:sentimentr /app
USER sentimentr

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "enhanced_sentimentr.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
