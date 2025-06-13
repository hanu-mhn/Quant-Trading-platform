# Quantitative Trading Platform Docker Image
# Developed by: Malavath Hanmanth Nayak
# Contact: hanmanthnayak.95@gmail.com
# LinkedIn: https://www.linkedin.com/in/hanmanth-nayak-m-6bbab1263/

FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory
WORKDIR /app

# Install system dependencies (including those required for TensorFlow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libffi-dev \
    libssl-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libreadline-dev \
    libx11-6 \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r quantuser && useradd -r -g quantuser quantuser

# Copy requirements for advanced builds
COPY requirements-full.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip==23.2.1 && \
    pip install --no-cache-dir -r requirements-full.txt

# Copy the entire application
COPY . .

# Change ownership to non-root user
RUN chown -R quantuser:quantuser /app

# Switch to non-root user
USER quantuser

# Expose ports for both API and Streamlit
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501 || exit 1

# Default command - run Streamlit dashboard
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]