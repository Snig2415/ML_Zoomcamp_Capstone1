FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src ./src
COPY models ./models

# Expose FastAPI port
EXPOSE 8000

# Start API
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
