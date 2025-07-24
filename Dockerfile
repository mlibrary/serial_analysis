# Use an official Python runtime as a parent image
FROM python:3.11

# Install system packages required for building Rust
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Set the working directory in the container
WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install PyPDF2

# Copy only the necessary files into the container at /usr/src/app
COPY *.py ./
COPY .env ./

# Run app.py when the container launches
CMD ["python", "./serial_analysis_PDF.py"]
