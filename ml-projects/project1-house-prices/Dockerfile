# Use an official lightweight Python runtime as a parent image
FROM python:3.11-slim

# Set system environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# PYTHONUNBUFFERED: Forces logs to stream straight to terminal without buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

# Set the working directory inside the container
WORKDIR /workspace

# Install system dependencies if required (slim images don't include build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt


# Copy the rest of the application files into the container
COPY ./app /workspace/app
COPY ./artifacts /workspace/artifacts

# Expose the internal port that Uvicorn will run on
EXPOSE 8000

# Run the application using Uvicorn bound to all network interfaces (0.0.0.0)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
