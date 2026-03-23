# 1) Base image: lightweight official Python
FROM python:3.11-slim

# 2) Ensure Python output is shown immediately in logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_NO_TF=1

WORKDIR /app

# 3) Install minimal system deps (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4) Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5) Create a non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# 6) Copy the app code
COPY api/ api/
COPY scripts/ scripts/
# Note: we DON'T copy vectorstore/ so we can mount it at runtime (keeps image small)

EXPOSE 8000

# 7) Start the server with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
