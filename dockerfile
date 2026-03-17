# -------- Base image --------
FROM python:3.10-slim

# -------- Environment --------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------- Working directory --------
WORKDIR /app

# -------- System dependencies --------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------- Install torch (CPU only, pinned) --------
RUN pip install --no-cache-dir \
    torch==1.13.1 \
    torchvision==0.14.1 \
    torchaudio==0.13.1 \
    --index-url https://download.pytorch.org/whl/cpu

# -------- Python dependencies --------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------- Application code --------
COPY src ./src
COPY main.py .
COPY .env .env

# -------- Expose FastAPI port --------
EXPOSE 8000

# -------- Start backend --------
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
