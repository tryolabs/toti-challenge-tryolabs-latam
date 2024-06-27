# syntax=docker/dockerfile:1.2
FROM python:3.9-slim

# Update system dependencies to avoid vulnerabilities
RUN apt-get update && apt-get upgrade -y

# Set the working directory
WORKDIR /delay_model

# Copy the requirements file
COPY requirements.txt .

# Install only the necessary dependencies to run the API
RUN pip install -r requirements.txt

# Copy python scripts and temp directory with the model checkpoint
# Avoid copying the exploration notebook
RUN mkdir challenge
COPY challenge/*.py challenge/
COPY challenge/tmp challenge/tmp

# Start the API
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "challenge:application"]