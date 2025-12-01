# Use a lightweight Python base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries directly (No requirements.txt needed)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    python-dotenv \
    google-generativeai \
    openai \
    qdrant-client \
    networkx \
    plyer \
    colorama \
    python-multipart

# Copy the application code
COPY . .

# Expose the port
EXPOSE 8080

# Command to run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]