#!/usr/bin/env bash

# Exit immediately if any underlying initialization step fails
set -e

# 1. Start the FastAPI backend pipeline explicitly using the python module flag
echo "Starting FastAPI backend pipeline on port 8000..."
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &

# 2. Extract Render's dynamic external routing port identifier (defaulting to 8501)
APP_PORT=${PORT:-8501}
echo "Starting Streamlit dashboard frontend UI on port $APP_PORT..."

# 3. Start Streamlit explicitly using the python module flag to bypass PATH issues
exec python -m streamlit run frontend.py --server.port="$APP_PORT" --server.address=0.0.0.0
