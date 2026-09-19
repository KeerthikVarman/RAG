# Use an official Python runtime as a base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PORT=8000 \
    HOST=0.0.0.0

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for C-extensions and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure data directories exist for PDF uploads and vector store persistence
RUN mkdir -p data/pdf data/vector_store

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Run the FastAPI server by default
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
