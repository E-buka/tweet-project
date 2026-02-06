FROM python:3.11-slim

# Install Java (required for PySpark)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

# Install Python dependencies first (Docker cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Render provides $PORT
CMD ["sh", "-c", "uvicorn tweet_analysis.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
