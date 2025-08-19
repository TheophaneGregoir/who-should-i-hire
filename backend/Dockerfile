FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Ensure python outputs are displayed during runtime
ENV PYTHONUNBUFFERED=1

ENV DATA_PATH=/data
ENV FAISS_TEXT_VECTOR_DB_PATH="/data/faiss_size_100000_text-embedding-3-large.index"
ENV FAISS_IMAGE_VECTOR_DB_PATH="/data/faiss_size_50000_CLIP.index"
ENV AMAZON_FASHION_FILE="/data/meta_Amazon_Fashion.jsonl"
ENV OPENAI_EMBEDDING_MODEL="text-embedding-3-large"

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

