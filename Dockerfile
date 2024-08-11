# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies and Poetry in one layer, then clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir poetry \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to cache them in docker layer
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Copy only necessary files from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy necessary application files
COPY ./rag_tender_export_fastapi ./rag_tender_export_fastapi
COPY ./data ./data
COPY main.py .
COPY rag_tender_export_fastapi/config.yml .

# Install only runtime dependencies
RUN pip install --no-cache-dir fastapi

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]