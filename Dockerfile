# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fraud_pipeline/ ./fraud_pipeline/
COPY scripts/ ./scripts/
COPY fraud_pipeline/configs/ ./fraud_pipeline/configs/
COPY pyproject.toml .
COPY setup.py .

# Install the package in development mode
RUN pip install -e .

# Create data directories
RUN mkdir -p data/raw data/processed data/models data/reports

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "fraud_pipeline.api:app", "--host", "0.0.0.0", "--port", "8000"]
