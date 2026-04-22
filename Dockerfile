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

# Update packages and install Poppler and Tesseract
RUN apt-get update && apt-get install -y \
poppler-utils \
tesseract-ocr \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# Install the Python wrappers
RUN pip install pytesseract pdf2image

# Copy only the necessary files into the container at /usr/src/app
COPY *.py ./

# Note: .env file is mounted at runtime for security (not copied into image)

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create and set permissions for mounted volume directory
RUN mkdir -p /input_and_output && chown -R appuser:appuser /input_and_output

# Change ownership of application directory
RUN chown -R appuser:appuser /usr/src/app

# Switch to non-root user
USER appuser

# Run app.py when the container launches
CMD ["python", "./serial_analysis_PDF.py"]
