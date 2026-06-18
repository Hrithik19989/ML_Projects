FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Streamlit only needs the frontend script
COPY frontend.py .

# Run Streamlit directly on the port provided by Render
CMD ["sh", "-c", "streamlit run frontend.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
