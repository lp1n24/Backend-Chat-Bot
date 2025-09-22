# Python 3.11 is the most stable version (I used 3.13.1 on my local)
FROM python:3.11-slim

# Tell system to not write .pyc files and show logs directly
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory inside container
WORKDIR /app

# Copy requirement list and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all project files into container
COPY src /app/src

# Start FastAPI with Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
