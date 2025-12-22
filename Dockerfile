
# Base Image
FROM python:3.10-slim

# Set Working Directory
WORKDIR /app

# Install System Dependencies (gcc for potential c++ extensions)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy Requirements
COPY requirements.txt .

# Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Source Code
COPY . .

# Set Python Path
ENV PYTHONPATH=/app

# Default Command (Overridden by Docker Compose)
CMD ["bash"]
