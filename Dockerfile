# Use an official Python runtime as the base image
FROM python:3.10-slim

# Create working dir
WORKDIR /app

# Install system deps (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy model artifacts and code
COPY ./anli_best_results /app/anli_best_results
COPY ./serve /app/serve

EXPOSE 8080

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
