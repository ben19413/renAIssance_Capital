# ------------------------------- 
# Base Stage: common setup 
# -------------------------------
    FROM python:3.11-slim AS base

    # Install system dependencies
    RUN apt-get update && apt-get install -y \
        git \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*
    
    # Set working directory and environment variables
    WORKDIR /app
    ENV PYTHONPATH=/app/src
    
    # -------------------------------
    # Production Stage: production image
    # -------------------------------
    FROM base
    
    # Copy necessary files
    COPY setup_production.py /app/setup.py
    COPY src/production/ /app/production/
    COPY production_requirements.txt /app/
    COPY src/config_production.json /app/production/
    COPY .env /app/.env
    
    # Install only production dependencies
    RUN pip install --no-cache-dir -r production_requirements.txt && pip install .
    
    # Load environment variables from .env file (you'll need to implement this in your application)
    ENV $(cat /app/.env | xargs)
    
    # Expose the port
    EXPOSE 8000
    
    # Set the container to auto-restart unless stopped
    LABEL com.docker.compose.restart_policy="unless-stopped"
    
    # Set the entrypoint
    CMD ["uvicorn", "production.server:app", "--host", "0.0.0.0", "--port", "8000"]