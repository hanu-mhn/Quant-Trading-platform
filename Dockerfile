FROM python:3.9-slim-bookworm

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
	pip install --no-cache-dir -r requirements.txt && \
	apt-get remove -y gcc && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the source code into the container
COPY src/ ./src/

# Set the entry point for the application
CMD ["python", "./src/live_trading/live_trader.py"]