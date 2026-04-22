from openai import AzureOpenAI
import openai
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from pathlib import Path
import time
import csv
import json
import signal
import re
import pytesseract # for OCR generation 
from pdf2image import convert_from_path # for OCR generation

# Security Configuration: File size and content limits
# These limits prevent DoS attacks while supporting legitimate large document processing
# Adjust these values based on your API limits and processing needs
#
# IMPORTANT: These are SAFETY limits, not API limits. They prevent:
#   - Processing malformed/malicious files that could crash the system
#   - Excessive API costs from accidentally processing huge files
#   - Denial of service from zip bombs or similar attacks
#
# Your API may support files up to 260k tokens. These limits are set generously
# to allow your full use case while still providing protection.
#
# To adjust limits, modify the values below:
MAX_FILE_SIZE_MB = 200  # Maximum file size in MB (allows large PDFs with images)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_TEXT_LENGTH = 1_200_000  # ~300k tokens at 4 chars/token (supports stated 260k token limit)
MAX_OCR_PAGES = 500  # Maximum pages to OCR PER FILE (not cumulative across all files)
OCR_TIMEOUT_SECONDS = 2400  # 40 minute timeout for OCR per file (~3 sec/page × 500 pages + buffer)
PDF_MAGIC_BYTES = b'%PDF'  # PDF file signature

# Rate limiting configuration
API_RETRY_ATTEMPTS = 3  # Number of retry attempts for API errors
API_RETRY_BASE_DELAY = 2  # Base delay in seconds for exponential backoff
API_RETRY_MAX_DELAY = 60  # Maximum delay in seconds between retries
API_REQUEST_DELAY = 1  # Minimum delay between successful requests (seconds)

#Sets the current working directory to be the same as the file.
os.chdir(os.path.dirname(os.path.abspath(__file__)))


#Load environment file for secrets.
try:
    if load_dotenv('.env') is False:
        raise TypeError
except TypeError:
    print('Unable to load .env file.')
    quit()


# Filename sanitization function (must be defined before validation functions)
def sanitize_filename(file_path):
    """Sanitize filename for safe logging and display.
    
    Removes or replaces characters that could be used for:
    - Path traversal attacks (../, ..\)
    - Command injection
    - Log injection (newlines, control characters)
    
    Returns only the base filename (not full path) with safe characters.
    """
    # Get just the filename, not the full path
    filename = file_path.name if hasattr(file_path, 'name') else str(file_path)
    
    # Remove any path separators (defense in depth)
    filename = filename.replace('/', '_').replace('\\', '_')
    
    # Remove control characters and newlines (prevent log injection)
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    
    # Remove or replace other potentially dangerous characters
    # Keep alphanumeric, spaces, dots, hyphens, underscores
    filename = re.sub(r'[^\w\s.\-]', '_', filename)
    
    # Limit length to prevent buffer overflow in logs
    if len(filename) > 255:
        filename = filename[:252] + '...'
    
    return filename


# Security validation functions
def validate_file_size(file_path):
    """Validate file size is within acceptable limits."""
    safe_name = sanitize_filename(file_path)
    file_size = file_path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        print(f"   SKIPPED: {safe_name} exceeds size limit ({file_size / 1024 / 1024:.1f}MB > {MAX_FILE_SIZE_MB}MB)", flush=True)
        return False
    if file_size == 0:
        print(f"   SKIPPED: {safe_name} is empty", flush=True)
        return False
    return True

def validate_pdf_format(file_path):
    """Verify file has valid PDF magic bytes."""
    safe_name = sanitize_filename(file_path)
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if not header.startswith(PDF_MAGIC_BYTES):
                print(f"   SKIPPED: {safe_name} is not a valid PDF file", flush=True)
                return False
        return True
    except Exception as e:
        print(f"   SKIPPED: {safe_name} - unable to read file", flush=True)
        return False

def validate_text_length(text, file_name):
    """Validate text length is within API token limits."""
    text_length = len(text)
    if text_length > MAX_TEXT_LENGTH:
        print(f"   SKIPPED: {file_name} text too long ({text_length:,} chars > {MAX_TEXT_LENGTH:,} chars limit)", flush=True)
        return False
    if text_length < 50:
        print(f"   SKIPPED: {file_name} has insufficient text ({text_length} chars)", flush=True)
        return False
    return True


def system_message():
    system_message = read_file_contents('/input_and_output/system_message.txt')
    return f"""
    {system_message}
    """

# Creates the assistant message for the api call.  The assistant message gives an example of how the LLM should respond.
def assistant_message():
    assistant_message = read_file_contents('/input_and_output/assistant_message.txt')

    return f"""

    {assistant_message}
    --"""


# Create usermessage function
def user_message(text):
    user_message = read_file_contents('/input_and_output/user_message.txt')
    return f"""
TASK:
        {user_message} Include the source_path "{source_path}" in your response.
    TEXT: {text}
    """

def read_file_contents(file_path):
    #Reads the contents of a text file given its file path.
    with open(file_path, "r") as file:
        contents = file.read()
    return contents

def is_binary_file(file_path, chunk_size=4096):
    with open(file_path, 'rb') as f:
        chunk = f.read(chunk_size)
    if b'\x00' in chunk:
        return True
    # Try multiple common encodings
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            chunk.decode(encoding)
            return False
        except (UnicodeDecodeError, LookupError):
            continue
    # If none of the common encodings work, consider it binary
    return True

def read_pdf_text(source_path):
    # Import all text (and nothing but text) from the PDF
    # 1. Attempt standard text extraction
    source_text = ""
    page_count = 0
    try:
        with open(source_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            page_count = len(pdf_reader.pages)
            
            # Check page count for OCR purposes
            if page_count > MAX_OCR_PAGES:
                safe_name = sanitize_filename(source_path)
                print(f"   WARNING: {safe_name} has {page_count} pages (max {MAX_OCR_PAGES} for OCR)", flush=True)
            
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    source_text += extracted
    except Exception as e:
        safe_name = sanitize_filename(source_path)
        print(f"   Error reading PDF structure for {safe_name}", flush=True)
        # Log detailed error for debugging (write to file in production)
        # print(f"   Debug: {type(e).__name__}")
        return ""

    # 2. Check if we actually got meaningful text
    # (Threshold of 50 chars ignores random metadata/page numbers)
    if len(source_text.strip()) < 50:
        safe_name = sanitize_filename(source_path)
        print(f"   {safe_name} appears to be a scan or has no text. Starting OCR...", flush=True)
        
        # Don't OCR files with too many pages (limit is PER FILE, not cumulative)
        if page_count > MAX_OCR_PAGES:
            print(f"   SKIPPED OCR: {safe_name} has too many pages ({page_count} > {MAX_OCR_PAGES})", flush=True)
            return ""
        
        try:
            # Set up timeout handler for OCR (timeout is PER FILE)
            def timeout_handler(signum, frame):
                raise TimeoutError(f"OCR exceeded {OCR_TIMEOUT_SECONDS} second timeout")
            
            # Only set alarm on Unix-like systems (not Windows)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(OCR_TIMEOUT_SECONDS)
            
            try:
                # Convert PDF to images
                images = convert_from_path(source_path, dpi=200)
                
                ocr_text = ""
                # Process up to MAX_OCR_PAGES pages from THIS file only
                for idx, image in enumerate(images):
                    if idx >= MAX_OCR_PAGES:
                        print(f"   Stopping OCR at page {MAX_OCR_PAGES}", flush=True)
                        break
                    ocr_text += pytesseract.image_to_string(image)
                
                return ocr_text
            finally:
                # Cancel the alarm
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                    
        except TimeoutError as timeout_err:
            safe_name = sanitize_filename(source_path)
            print(f"   OCR timeout for {safe_name} (exceeded {OCR_TIMEOUT_SECONDS}s limit)", flush=True)
            return ""
        except Exception as ocr_err:
            safe_name = sanitize_filename(source_path)
            print(f"   OCR failed for {safe_name}", flush=True)
            return "" # Return empty so the loop skip logic triggers
    
    return source_text

def read_text_file(source_path):
    """Read text file with encoding fallback and size limits."""
    safe_name = sanitize_filename(source_path)
    
    try:
        # Read with size limit (don't load entire file if too large)
        file_size = source_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            print(f"   SKIPPED: {safe_name} exceeds size limit", flush=True)
            return None
            
        with open(source_path, 'r', encoding='utf-8') as txt_file:
            return txt_file.read()
    except UnicodeDecodeError:
        print(f"   Warning: {safe_name} has encoding issues (attempting recovery)", flush=True)
        try:
            with open(source_path, 'r', encoding='utf-8', errors='ignore') as txt_file:
                return txt_file.read()
        except Exception as e:
            print(f"   SKIPPED: {safe_name} - unable to read file", flush=True)
            return None
    except Exception as e:
        print(f"   SKIPPED: {safe_name} - unable to read file", flush=True)
        return None

# Helper function to recursively flatten lists and nested dictionaries into a single string
def normalize_value(value):
    """Recursively flattens lists and dictionaries into a single comma-separated string, 
    including keys for dictionary values."""
    
    if isinstance(value, list):
        # If it's a list, recursively process each item and join them.
        processed_items = [normalize_value(item) for item in value]
        # Use filter(None, ...) to safely remove any empty strings resulting from recursion
        return ", ".join(filter(None, processed_items))
    
    elif isinstance(value, dict):
        # If it's a dictionary, we process key-value pairs to preserve context.
        processed_pairs = []
        for key, v in value.items():
            # Recursively process the value (v) first
            normalized_v = normalize_value(v)
            
            # If the normalized value is not empty, join the key and value with a colon and space
            if normalized_v:
                processed_pairs.append(f"{key}: {normalized_v}")
                
        return ", ".join(processed_pairs)
    
    elif value is None or value == "":
        # Handle null or empty string values gracefully
        return ""
    
    else:
        # Base case: return strings, numbers, etc. as-is
        return str(value)


def process_source(source_path, source_text, client, fieldnames, csv_writer):
    """Process a single source file with the API, including retry logic and rate limiting."""
    
    safe_filename = sanitize_filename(source_path)
    
    # Create Query
    messages = [
        {"role": "system", "content": system_message()},
        {"role": "assistant", "content": assistant_message()},
        {"role": "user", "content": user_message(text=source_text)}
    ]

    # Implement retry logic with exponential backoff
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            # Make the call to the UMGPT Toolkit Azure API
            response = client.chat.completions.create(
                model=os.environ['MODEL'],
                messages=messages,
                temperature=0.0,
                response_format={ "type": "json_object" }
            )
            
            # Success - break out of retry loop
            break
            
        except openai.RateLimitError as e:
            # Handle rate limiting with exponential backoff
            if attempt < API_RETRY_ATTEMPTS - 1:
                delay = min(API_RETRY_BASE_DELAY * (2 ** attempt), API_RETRY_MAX_DELAY)
                print(f"   Rate limit reached for {safe_filename}. Retrying in {delay}s (attempt {attempt + 1}/{API_RETRY_ATTEMPTS})...", flush=True)
                time.sleep(delay)
            else:
                print(f"   SKIPPED: {safe_filename} - Rate limit exceeded after {API_RETRY_ATTEMPTS} attempts", flush=True)
                return
                
        except openai.APITimeoutError as e:
            # Handle timeout errors
            if attempt < API_RETRY_ATTEMPTS - 1:
                delay = min(API_RETRY_BASE_DELAY * (2 ** attempt), API_RETRY_MAX_DELAY)
                print(f"   API timeout for {safe_filename}. Retrying in {delay}s (attempt {attempt + 1}/{API_RETRY_ATTEMPTS})...", flush=True)
                time.sleep(delay)
            else:
                print(f"   SKIPPED: {safe_filename} - API timeout after {API_RETRY_ATTEMPTS} attempts", flush=True)
                return
                
        except openai.APIConnectionError as e:
            # Handle connection errors
            if attempt < API_RETRY_ATTEMPTS - 1:
                delay = min(API_RETRY_BASE_DELAY * (2 ** attempt), API_RETRY_MAX_DELAY)
                print(f"   Connection error for {safe_filename}. Retrying in {delay}s (attempt {attempt + 1}/{API_RETRY_ATTEMPTS})...", flush=True)
                time.sleep(delay)
            else:
                print(f"   SKIPPED: {safe_filename} - Connection failed after {API_RETRY_ATTEMPTS} attempts", flush=True)
                return
                
        except openai.AuthenticationError as e:
            # Don't retry authentication errors
            print(f"   SKIPPED: {safe_filename} - Authentication failed. Check API key.", flush=True)
            return
            
        except Exception as e:
            # Handle other API errors
            if attempt < API_RETRY_ATTEMPTS - 1:
                delay = min(API_RETRY_BASE_DELAY * (2 ** attempt), API_RETRY_MAX_DELAY)
                print(f"   API error for {safe_filename}. Retrying in {delay}s (attempt {attempt + 1}/{API_RETRY_ATTEMPTS})...", flush=True)
                time.sleep(delay)
            else:
                print(f"   SKIPPED: {safe_filename} - API request failed after {API_RETRY_ATTEMPTS} attempts", flush=True)
                return

    # Parse JSON UMGPT Toolkit response
    try:
        json_response = response.choices[0].message.content
        data = json.loads(json_response)

        # Write to CSV
        row_data = {}
        for fieldname in fieldnames:
            value = data.get(fieldname)
            row_data[fieldname] = normalize_value(value)

        csv_writer.writerow(row_data)
        
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        print(f"   SKIPPED: {safe_filename} - Invalid API response format", flush=True)
        return
    except Exception as e:
        print(f"   SKIPPED: {safe_filename} - Error processing response", flush=True)
        return

    # Rate limiting: Wait between successful requests
    time.sleep(API_REQUEST_DELAY)


# Directory to read PDFs
pdf_directory = '/input_and_output/PDFs'
pdf_directory_path = Path(pdf_directory)

# Directory to read TXT (and any non-binary files regardless of extension)
txt_directory = '/input_and_output/TXT'
txt_directory_path = Path(txt_directory)

# Path to save the CSV file
csv_file_path = '/input_and_output/extracted_data.csv'

# fieldnames (aka Column headers for the CSV output)
# the list of fieldnames should match the list of fieldnames in the assistant_message JSON example
fieldnames = read_file_contents('/input_and_output/fieldnames.txt').splitlines()

# Create Azure client once (more efficient than recreating per file)
client = AzureOpenAI(
    api_key=os.environ['OPENAI_API_KEY'],  
    api_version=os.environ['API_VERSION'],
    azure_endpoint=os.environ['OPENAI_API_BASE'],
    organization=os.environ['OPENAI_ORGANIZATION']
)

# Open CSV file for writing
with open(csv_file_path, 'w', newline='') as csv_file:
    # Define CSV writer
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write the header
    csv_writer.writeheader()

    # Process PDFs (sorted alphanumerically by filename)
    for source_path in sorted(pdf_directory_path.glob('*.pdf'), key=lambda p: p.name):
        print(source_path.name, flush=True)
        
        # Validate file size
        if not validate_file_size(source_path):
            continue
        
        # Validate PDF format
        if not validate_pdf_format(source_path):
            continue
        
        source_text = read_pdf_text(source_path)

        # Skip if no text found even after OCR attempt
        if not source_text or not source_text.strip():
            print(f"   SKIPPED: {source_path.name} - No text could be extracted.", flush=True)
            continue
        
        # Validate text length
        if not validate_text_length(source_text, source_path.name):
            continue
        
        process_source(source_path, source_text, client, fieldnames, csv_writer)

    # Process non-binary files under TXT (any extension), recursively (sorted alphanumerically by filename)
    if txt_directory_path.exists():
        txt_files = [p for p in txt_directory_path.rglob('*') if p.is_file() and not is_binary_file(p)]
        print(f"Found {len(txt_files)} text files to process", flush=True)
        for source_path in sorted(txt_files, key=lambda p: p.name):
            print(source_path.name, flush=True)
            
            # Validate file size (already done in read_text_file, but checking here too)
            if not validate_file_size(source_path):
                continue
            
            source_text = read_text_file(source_path)
            
            # Skip if file couldn't be read
            if source_text is None:
                continue
                
            # Validate text length
            if not validate_text_length(source_text, source_path.name):
                continue
            
            process_source(source_path, source_text, client, fieldnames, csv_writer)
    else:
        print(f"TXT directory not found at {txt_directory_path}", flush=True)

print(f"Data successfully written to {csv_file_path}")

# Print the CSV contents
#print_csv_contents(csv_file_path)

# Function to read and print CSV contents
def print_csv_contents(file_path):
    with open(file_path, 'r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            print(row)