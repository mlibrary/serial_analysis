from openai import AzureOpenAI
import openai
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from pathlib import Path
import time
import csv
import json
import pytesseract # for OCR generation 
from pdf2image import convert_from_path # for OCR generation

#Sets the current working directory to be the same as the file.
os.chdir(os.path.dirname(os.path.abspath(__file__)))


#Load environment file for secrets.
try:
    if load_dotenv('.env') is False:
        raise TypeError
except TypeError:
    print('Unable to load .env file.')
    quit()


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
    try:
        with open(source_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    source_text += extracted
    except Exception as e:
        print(f"   Error reading PDF structure for {source_path.name}: {e}")

    # 2. Check if we actually got meaningful text
    # (Threshold of 50 chars ignores random metadata/page numbers)
    if len(source_text.strip()) < 50:
        print(f"   {source_path.name} appears to be a scan or has no text. Starting OCR...")
        try:
            # Convert PDF to images
            images = convert_from_path(source_path, dpi=200)
            
            ocr_text = ""
            for image in images:
                ocr_text += pytesseract.image_to_string(image)
            
            return ocr_text
        except Exception as ocr_err:
            print(f"   OCR failed for {source_path.name}: {ocr_err}")
            return "" # Return empty so the loop skip logic triggers
    
    return source_text

def read_text_file(source_path):
    try:
        with open(source_path, 'r', encoding='utf-8') as txt_file:
            return txt_file.read()
    except UnicodeDecodeError:
        print(f"Warning: UnicodeDecodeError in file (skipping bad characters): {source_path}", flush=True)
        with open(source_path, 'r', encoding='utf-8', errors='ignore') as txt_file:
            return txt_file.read()

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
    # Create Query
    messages = [
        {"role": "system", "content": system_message()},
        {"role": "assistant", "content": assistant_message()},
        {"role": "user", "content": user_message(text=source_text)}
    ]

    # Make the call to the UMGPT Toolkit Azure API
    try:
    	response = client.chat.completions.create(
        	model=os.environ['MODEL'],
        	messages=messages,
        	temperature=0.0,
        	response_format={ "type": "json_object" }
    	)
    except Exception as e:
        print(f"!!! Error on {pdf_path.name}: {e}")
        # This 'continue' tells Python to skip the rest of the loop 
        # for THIS file and move to the next PDF instead of crashing.
        return

    # Parse JSON UMGPT Toolkit response
    json_response = response.choices[0].message.content

    try:
        # Assumes the JSON response is correctly formatted,
        # which depends on a good example in the assistant_message, and UMGPT doing it well
        data = json.loads(json_response)

        # --- UPDATED CSV WRITING LOGIC ---
        row_data = {}
        for fieldname in fieldnames:
            value = data.get(fieldname)
            # Use the helper function to ensure the value is a plain string
            row_data[fieldname] = normalize_value(value)

        csv_writer.writerow(row_data)
        # ---------------------------------

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

    # Wait X seconds between requests our of respect for the API service and to avoid throttling
    # Maybe not necessary if each API call takes more than a few seconds, 
    # which depends in part on how large the files are: 
    time.sleep(1)


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
        source_text = read_pdf_text(source_path)

        # Skip if no text found even after OCR attempt
        if not source_text.strip():
            print(f"   Skipping {source_path.name}: No text could be extracted.")
            continue
        
        process_source(source_path, source_text, client, fieldnames, csv_writer)

    # Process non-binary files under TXT (any extension), recursively (sorted alphanumerically by filename)
    if txt_directory_path.exists():
        txt_files = [p for p in txt_directory_path.rglob('*') if p.is_file() and not is_binary_file(p)]
        print(f"Found {len(txt_files)} text files to process", flush=True)
        for source_path in sorted(txt_files, key=lambda p: p.name):
            print(source_path.name, flush=True)
            source_text = read_text_file(source_path)
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