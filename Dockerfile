
FROM python:3.11-slim

WORKDIR /app

# ── Install system dependencies needed by PyMuPDF ─────────────────
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Copy and install Python dependencies first (layer caching) ────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy the rest of the application ──────────────────────────────
COPY app.py .
COPY src/ ./src/

# ── Expose Streamlit's default port ───────────────────────────────
EXPOSE 8501

# ── Tell Streamlit not to open a browser inside the container ─────
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

# ── Command that runs when the container starts ───────────────────
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]