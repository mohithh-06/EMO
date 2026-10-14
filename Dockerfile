# Multi-stage Dockerfile for emo - Fashion AI Advisor
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8501

WORKDIR /app

# Install system dependencies for OpenCV/Pillow and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python packages (using CPU torch wheels for lightweight cloud deployment)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Healthcheck for container orchestrators (Render, Cloud Run, Hugging Face)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit app
ENTRYPOINT ["streamlit", "run", "fitcheck/Fashion AI Advisor.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
