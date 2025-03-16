# Use an official Python image with FastAPI support
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Copy your application code to the container
COPY . /app

# Expose the port (the base image defaults to port 80)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
