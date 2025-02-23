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
    
    
    # Copy common files
    COPY setup.py .
    COPY src/ /app/src/
    
    # -------------------------------
    # Production Stage: production image
    # -------------------------------
    FROM base AS production
    
    # Copy production-specific requirements file
    COPY src/production/app /app/
    COPY production_requirements.txt /
    EXPOSE 8000
    # Install only production dependencies

    RUN pip install --no-cache-dir -r ../production_requirements.txt && pip install .
    
    # Replace with your production entrypoint as needed
    CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
    
    # -------------------------------
    # Development Stage: development image
    # -------------------------------
    FROM base AS development
    COPY setup.py .
    COPY src/ /app/src/
    # Copy development requirements file
    COPY requirements.txt /app/
    EXPOSE 8888
    # Install all dependencies needed for development
    RUN pip install --no-cache-dir -r requirements.txt && pip install .
    
    # For development, you might want to run an interactive shell or use a debugger;
    # here we use a placeholder command.
    CMD ["tail", "-f", "/dev/null"]
    


