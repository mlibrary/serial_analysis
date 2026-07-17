from openai import OpenAI
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
import keyword
import pytesseract  # for OCR generation
from pdf2image import convert_from_path  # for OCR generation
from pydantic import Field, ConfigDict, create_model, ValidationError


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


# Sets the current working directory to be the same as the file.
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Load environment file for secrets.
try:
    if load_dotenv('.env') is False:
        raise TypeError
except TypeError:
    print('Unable to load .env file.')
    quit()


# Filename sanitization function must be defined before validation functions.
def sanitize_filename(file_path):
    """Sanitize filename for safe logging and display.

    Removes or replaces characters that could be used for:
    - Path traversal attacks (../, ..\)
    - Command injection
    - Log injection (newlines, control characters)

    Returns only the base filename, not full path, with safe characters.
    """
    # Get just the filename, not the full path.
    filename = file_path.name if hasattr(file_path, 'name') else str(file_path)

    # Remove any path separators, defense in depth.
    filename = filename.replace('/', '_').replace('\\', '_')

    # Remove control characters and newlines, preventing log injection.
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)

    # Remove or replace other potentially dangerous characters.
    # Keep alphanumeric, spaces, dots, hyphens, underscores.
    filename = re.sub(r'[^\w\s.\-]', '_', filename)

    # Limit length to prevent buffer overflow in logs.
    if len(filename) > 255:
        filename = filename[:252] + '...'

    return filename


# Security validation functions.
def validate_file_size(file_path):
    """Validate file size is within acceptable limits."""
    safe_name = sanitize_filename(file_path)
    file_size = file_path.stat().st_size

    if file_size > MAX_FILE_SIZE_BYTES:
        print(
            f"   SKIPPED: {safe_name} exceeds size limit "
            f"({file_size / 1024 / 1024:.1f}MB > {MAX_FILE_SIZE_MB}MB)",
            flush=True
        )
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

    except Exception:
        print(f"   SKIPPED: {safe_name} - unable to read file", flush=True)
        return False


def validate_text_length(text, file_name):
    """Validate text length is within API token limits."""
    text_length = len(text)

    if text_length > MAX_TEXT_LENGTH:
        print(
            f"   SKIPPED: {file_name} text too long "
            f"({text_length:,} chars > {MAX_TEXT_LENGTH:,} chars limit)",
            flush=True
        )
        return False

    if text_length < 50:
        print(f"   SKIPPED: {file_name} has insufficient text ({text_length} chars)", flush=True)
        return False

    return True


def read_file_contents(file_path):
    """Reads the contents of a text file given its file path."""
    with open(file_path, "r", encoding="utf-8") as file:
        contents = file.read()
    return contents


def system_message():
    system_message_text = read_file_contents('/input_and_output/system_message.txt')
    return f"""
    {system_message_text}
    """


# Creates the assistant message for the API call.
# The assistant message gives an example of how the LLM should respond.
def assistant_message():
    assistant_message_text = read_file_contents('/input_and_output/assistant_message.txt')

    return f"""

    {assistant_message_text}
    --"""


# Create user message function.
def user_message(text, source_path, fieldnames):
    user_message_text = read_file_contents('/input_and_output/user_message.txt')

    source_path_instruction = ""
    if "source_path" in fieldnames:
        source_path_instruction = f' Include the source_path "{source_path}" in your response.'

    return f"""
TASK:
        {user_message_text}{source_path_instruction}
    TEXT: {text}
    """


def is_binary_file(file_path, chunk_size=4096):
    with open(file_path, 'rb') as f:
        chunk = f.read(chunk_size)

    if b'\x00' in chunk:
        return True

    # Try multiple common encodings.
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            chunk.decode(encoding)
            return False
        except (UnicodeDecodeError, LookupError):
            continue

    # If none of the common encodings work, consider it binary.
    return True


def read_pdf_text(source_path):
    """Import all text from the PDF, using OCR if standard extraction fails."""

    # 1. Attempt standard text extraction.
    source_text = ""
    page_count = 0

    try:
        with open(source_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            page_count = len(pdf_reader.pages)

            # Check page count for OCR purposes.
            if page_count > MAX_OCR_PAGES:
                safe_name = sanitize_filename(source_path)
                print(
                    f"   WARNING: {safe_name} has {page_count} pages "
                    f"(max {MAX_OCR_PAGES} for OCR)",
                    flush=True
                )

            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    source_text += extracted

    except Exception:
        safe_name = sanitize_filename(source_path)
        print(f"   Error reading PDF structure for {safe_name}", flush=True)
        return ""

    # 2. Check if we actually got meaningful text.
    # Threshold of 50 chars ignores random metadata/page numbers.
    if len(source_text.strip()) < 50:
        safe_name = sanitize_filename(source_path)
        print(f"   {safe_name} appears to be a scan or has no text. Starting OCR...", flush=True)

        # Don't OCR files with too many pages.
        if page_count > MAX_OCR_PAGES:
            print(
                f"   SKIPPED OCR: {safe_name} has too many pages "
                f"({page_count} > {MAX_OCR_PAGES})",
                flush=True
            )
            return ""

        try:
            # Set up timeout handler for OCR.
            def timeout_handler(signum, frame):
                raise TimeoutError(f"OCR exceeded {OCR_TIMEOUT_SECONDS} second timeout")

            # Only set alarm on Unix-like systems, not Windows.
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(OCR_TIMEOUT_SECONDS)

            try:
                # Convert PDF to images.
                images = convert_from_path(source_path, dpi=200)

                ocr_text = ""

                # Process up to MAX_OCR_PAGES pages from THIS file only.
                for idx, image in enumerate(images):
                    if idx >= MAX_OCR_PAGES:
                        print(f"   Stopping OCR at page {MAX_OCR_PAGES}", flush=True)
                        break

                    ocr_text += pytesseract.image_to_string(image)

                return ocr_text

            finally:
                # Cancel the alarm.
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)

        except TimeoutError:
            safe_name = sanitize_filename(source_path)
            print(
                f"   OCR timeout for {safe_name} "
                f"(exceeded {OCR_TIMEOUT_SECONDS}s limit)",
                flush=True
            )
            return ""

        except Exception:
            safe_name = sanitize_filename(source_path)
            print(f"   OCR failed for {safe_name}", flush=True)
            return ""

    return source_text


def read_text_file(source_path):
    """Read text file with encoding fallback and size limits."""
    safe_name = sanitize_filename(source_path)

    try:
        # Read with size limit.
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

        except Exception:
            print(f"   SKIPPED: {safe_name} - unable to read file", flush=True)
            return None

    except Exception:
        print(f"   SKIPPED: {safe_name} - unable to read file", flush=True)
        return None


# Helper function to recursively flatten lists and nested dictionaries into a single string.
def normalize_value(value):
    """Recursively flattens lists and dictionaries into a single comma-separated string.

    Even though the structured schema requires strings, this function is retained
    as a defensive safeguard before writing CSV rows.
    """

    if isinstance(value, list):
        processed_items = [normalize_value(item) for item in value]
        return ", ".join(filter(None, processed_items))

    elif isinstance(value, dict):
        processed_pairs = []

        for key, v in value.items():
            normalized_v = normalize_value(v)

            if normalized_v:
                processed_pairs.append(f"{key}: {normalized_v}")

        return ", ".join(processed_pairs)

    elif value is None or value == "":
        return ""

    else:
        return str(value)


def build_extract_model_from_fieldnames(fieldnames):
    """
    Dynamically create a Pydantic model whose JSON field names exactly match
    the CSV column headers from fieldnames.txt.

    Each field is required and must be a string.
    """
    cleaned_fieldnames = [field.strip() for field in fieldnames if field.strip()]

    duplicates = sorted({
        field for field in cleaned_fieldnames
        if cleaned_fieldnames.count(field) > 1
    })

    if duplicates:
        raise ValueError(f"Duplicate fieldnames found: {duplicates}")

    used_internal_names = set()
    model_fields = {}

    for header in cleaned_fieldnames:
        # Create a safe internal Python field name.
        # The JSON output will still use the original CSV header via alias=header.
        internal_name = re.sub(r'\W', '_', header)

        if not internal_name:
            internal_name = "field"

        if re.match(r'^\d', internal_name):
            internal_name = f"field_{internal_name}"

        if keyword.iskeyword(internal_name):
            internal_name = f"{internal_name}_"

        # Ensure internal field names are unique even if sanitized names collide.
        base_name = internal_name
        counter = 2

        while internal_name in used_internal_names:
            internal_name = f"{base_name}_{counter}"
            counter += 1

        used_internal_names.add(internal_name)

        # Required string field.
        # alias=header makes the JSON Schema property name match the CSV header.
        model_fields[internal_name] = (
            str,
            Field(..., alias=header)
        )

    ExtractModel = create_model(
        "Extract",
        __config__=ConfigDict(
            populate_by_name=True,
            extra="forbid"
        ),
        **model_fields
    )

    return ExtractModel


def print_api_error_details(error, safe_filename):
    """Print detailed API error information for debugging."""
    print(f"\n--- API ERROR DETAILS for {safe_filename} ---", flush=True)
    print(f"Error type: {type(error).__name__}", flush=True)
    print(f"Error repr: {repr(error)}", flush=True)
    print(f"Error str: {str(error)}", flush=True)

    # OpenAI SDK errors often expose these attributes.
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        print(f"Status code: {status_code}", flush=True)

    code = getattr(error, "code", None)
    if code is not None:
        print(f"Error code: {code}", flush=True)

    param = getattr(error, "param", None)
    if param is not None:
        print(f"Error param: {param}", flush=True)

    body = getattr(error, "body", None)
    if body is not None:
        print(f"Error body: {body}", flush=True)

    response = getattr(error, "response", None)
    if response is not None:
        try:
            print(f"Response status code: {response.status_code}", flush=True)
            print(f"Response text: {response.text}", flush=True)
        except Exception as response_print_error:
            print(f"Unable to print raw response: {repr(response_print_error)}", flush=True)

    print("--- END API ERROR DETAILS ---\n", flush=True)


def process_source(source_path, source_text, client, fieldnames, csv_writer, ExtractModel, extract_schema):
    """Process a single source file with the API, including retry logic and rate limiting."""

    safe_filename = sanitize_filename(source_path)

    instructions = f"""
{system_message()}

{assistant_message()}

Return exactly one JSON object matching the supplied JSON Schema.

Important:
- Every field in the schema is required.
- Every value must be a string.
- If a value cannot be determined, return an empty string for that field.
- Do not include any fields that are not in the schema.
"""

    input_text = user_message(
        text=source_text,
        source_path=str(source_path),
        fieldnames=fieldnames
    )

    # Implement retry logic with exponential backoff.
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            # Make the call to the UMGPT Toolkit/OpenAI-compatible Responses API.
            response = client.responses.create(
                model=os.environ['MODEL'],
                instructions=instructions,
                input=input_text,
                temperature=0.0,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "Extract",
                        "schema": extract_schema,
                        "strict": True,
                    }
                },
            )

            # Success - break out of retry loop.
            break

        except openai.RateLimitError as e:
            print_api_error_details(e, safe_filename)

            if attempt < API_RETRY_ATTEMPTS - 1:
                delay = min(API_RETRY_BASE_DELAY * (2 ** attempt), API_RETRY_MAX_DELAY)
                print(
                    f"   Rate limit reached for {safe_filename}. "
                    f"Retrying in {delay}s "
                    f"(attempt {attempt + 1}/{API_RETRY_ATTEMPTS})...",
                    flush=True
                )
                time.sleep(delay)
            else:
                print(
                    f"   SKIPPED: {safe_filename} - "
                    f"Rate limit exceeded after {API_RETRY_ATTEMPTS} attempts",
                    flush=True
                )
                return

        except openai.APITimeoutError as e:
            print_api_error_details(e, safe_filename)

            if attempt < API_RETRY_ATTEMPTS - 1:
                delay = min(API_RETRY_BASE_DELAY * (2 ** attempt), API_RETRY_MAX_DELAY)
                print(
                    f"   API timeout for {safe_filename}. "
                    f"Retrying in {delay}s "
                    f"(attempt {attempt + 1}/{API_RETRY_ATTEMPTS})...",
                    flush=True
                )
                time.sleep(delay)
            else:
                print(
                    f"   SKIPPED: {safe_filename} - "
                    f"API timeout after {API_RETRY_ATTEMPTS} attempts",
                    flush=True
                )
                return

        except openai.APIConnectionError as e:
            print_api_error_details(e, safe_filename)

            if attempt < API_RETRY_ATTEMPTS - 1:
                delay = min(API_RETRY_BASE_DELAY * (2 ** attempt), API_RETRY_MAX_DELAY)
                print(
                    f"   Connection error for {safe_filename}. "
                    f"Retrying in {delay}s "
                    f"(attempt {attempt + 1}/{API_RETRY_ATTEMPTS})...",
                    flush=True
                )
                time.sleep(delay)
            else:
                print(
                    f"   SKIPPED: {safe_filename} - "
                    f"Connection failed after {API_RETRY_ATTEMPTS} attempts",
                    flush=True
                )
                return

        except openai.AuthenticationError as e:
            print_api_error_details(e, safe_filename)

            print(
                f"   SKIPPED: {safe_filename} - "
                f"Authentication failed. Check API key.",
                flush=True
            )
            return

        except openai.BadRequestError as e:
            print_api_error_details(e, safe_filename)

            print(
                f"   SKIPPED: {safe_filename} - "
                f"Bad request. This is often caused by an invalid schema, unsupported parameter, "
                f"too much input text, or a model that does not support structured outputs.",
                flush=True
            )
            return

        except openai.APIStatusError as e:
            print_api_error_details(e, safe_filename)

            if attempt < API_RETRY_ATTEMPTS - 1:
                delay = min(API_RETRY_BASE_DELAY * (2 ** attempt), API_RETRY_MAX_DELAY)
                print(
                    f"   API status error for {safe_filename}. "
                    f"Retrying in {delay}s "
                    f"(attempt {attempt + 1}/{API_RETRY_ATTEMPTS})...",
                    flush=True
                )
                time.sleep(delay)
            else:
                print(
                    f"   SKIPPED: {safe_filename} - "
                    f"API status error after {API_RETRY_ATTEMPTS} attempts",
                    flush=True
                )
                return

        except Exception as e:
            print_api_error_details(e, safe_filename)

            if attempt < API_RETRY_ATTEMPTS - 1:
                delay = min(API_RETRY_BASE_DELAY * (2 ** attempt), API_RETRY_MAX_DELAY)
                print(
                    f"   API error for {safe_filename}. "
                    f"Retrying in {delay}s "
                    f"(attempt {attempt + 1}/{API_RETRY_ATTEMPTS})...",
                    flush=True
                )
                time.sleep(delay)
            else:
                print(
                    f"   SKIPPED: {safe_filename} - "
                    f"API request failed after {API_RETRY_ATTEMPTS} attempts",
                    flush=True
                )
                return

    # Validate structured JSON response with Pydantic.
    try:
        result = ExtractModel.model_validate_json(response.output_text)

        # Dump using aliases so keys exactly match CSV fieldnames.
        data = result.model_dump(by_alias=True)

        # Optional safeguard: if source_path is one of your CSV fields, make it deterministic.
        if "source_path" in data:
            data["source_path"] = str(source_path)

        # Write to CSV.
        row_data = {}

        for fieldname in fieldnames:
            value = data.get(fieldname, "")
            row_data[fieldname] = normalize_value(value)

        csv_writer.writerow(row_data)

    except ValidationError as e:
        print(
            f"   SKIPPED: {safe_filename} - "
            f"API response did not match the required JSON schema",
            flush=True
        )
        print(f"   Validation error details: {repr(e)}", flush=True)
        return

    except AttributeError as e:
        print(
            f"   SKIPPED: {safe_filename} - "
            f"Invalid Responses API output format",
            flush=True
        )
        print(f"   Attribute error details: {repr(e)}", flush=True)
        return

    except Exception as e:
        print(
            f"   SKIPPED: {safe_filename} - "
            f"Error processing structured response",
            flush=True
        )
        print(f"   Processing error details: {repr(e)}", flush=True)
        return

    # Rate limiting: Wait between successful requests.
    time.sleep(API_REQUEST_DELAY)


# Directory to read PDFs.
pdf_directory = '/input_and_output/PDFs'
pdf_directory_path = Path(pdf_directory)

# Directory to read TXT and any non-binary files regardless of extension.
txt_directory = '/input_and_output/TXT'
txt_directory_path = Path(txt_directory)

# Path to save the CSV file.
csv_file_path = '/input_and_output/extracted_data.csv'

# fieldnames, aka column headers for the CSV output.
# The list of fieldnames defines the structured JSON response schema.
fieldnames = [
    field.strip()
    for field in read_file_contents('/input_and_output/fieldnames.txt').splitlines()
    if field.strip()
]

# Build the dynamic Pydantic model from the CSV headers.
ExtractModel = build_extract_model_from_fieldnames(fieldnames)

# Generate JSON Schema using the CSV headers as JSON property names.
extract_schema = ExtractModel.model_json_schema(by_alias=True)

# Required for strict JSON Schema mode.
extract_schema["additionalProperties"] = False

# Print generated JSON Schema once for debugging.
print("Generated JSON Schema:", flush=True)
print(json.dumps(extract_schema, indent=2), flush=True)

# Create OpenAI-compatible client once.
# This follows the Responses API example and uses OPENAI_API_BASE as base_url.
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    base_url=os.environ['OPENAI_API_BASE'],
    organization=os.environ.get('OPENAI_ORGANIZATION') or None,
)


# Open CSV file for writing.
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    # Define CSV writer.
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header.
    csv_writer.writeheader()

    # Process PDFs, sorted alphanumerically by filename.
    for source_path in sorted(pdf_directory_path.glob('*.pdf'), key=lambda p: p.name):
        print(source_path.name, flush=True)

        # Validate file size.
        if not validate_file_size(source_path):
            continue

        # Validate PDF format.
        if not validate_pdf_format(source_path):
            continue

        source_text = read_pdf_text(source_path)

        # Skip if no text found even after OCR attempt.
        if not source_text or not source_text.strip():
            print(f"   SKIPPED: {source_path.name} - No text could be extracted.", flush=True)
            continue

        # Validate text length.
        if not validate_text_length(source_text, source_path.name):
            continue

        process_source(
            source_path=source_path,
            source_text=source_text,
            client=client,
            fieldnames=fieldnames,
            csv_writer=csv_writer,
            ExtractModel=ExtractModel,
            extract_schema=extract_schema
        )

    # Process non-binary files under TXT, any extension, recursively.
    if txt_directory_path.exists():
        txt_files = [
            p for p in txt_directory_path.rglob('*')
            if p.is_file() and not is_binary_file(p)
        ]

        print(f"Found {len(txt_files)} text files to process", flush=True)

        for source_path in sorted(txt_files, key=lambda p: p.name):
            print(source_path.name, flush=True)

            # Validate file size.
            if not validate_file_size(source_path):
                continue

            source_text = read_text_file(source_path)

            # Skip if file could not be read.
            if source_text is None:
                continue

            # Validate text length.
            if not validate_text_length(source_text, source_path.name):
                continue

            process_source(
                source_path=source_path,
                source_text=source_text,
                client=client,
                fieldnames=fieldnames,
                csv_writer=csv_writer,
                ExtractModel=ExtractModel,
                extract_schema=extract_schema
            )

    else:
        print(f"TXT directory not found at {txt_directory_path}", flush=True)


print(f"Data successfully written to {csv_file_path}")


# Function to read and print CSV contents.
def print_csv_contents(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            print(row)


# Print the CSV contents if desired.
# print_csv_contents(csv_file_path)