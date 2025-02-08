# Use a minimal Python 3.11 image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \ 
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone public repository FIRST
COPY setup.py .
COPY src  /app/src

ENV PYTHONPATH=/app/src

# Copy only additional necessary files
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install .

# Command to run your script
CMD ["tail", "-f", "/dev/null"]


