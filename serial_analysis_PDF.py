from openai import AzureOpenAI
import openai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pathlib import Path
import time
import csv
import json

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
    system_message=read_file_contents('/input_and_output/system_message.txt')
    return f"""
    {system_message}
    """

# Creates the assistant message for the api call.  The assistant message gives an example of how the LLM should respond.
def assistant_message():
    assistant_message=read_file_contents('/input_and_output/assistant_message.txt')

    return f"""

    {assistant_message}
 


# Create usermessage function
def user_message(text):
    user_message=read_file_contents('/input_and_output/user_message.txt')
    return f"""
TASK:
        {user_message} Include the pdf_path "{pdf_path}" in your response.
    TEXT: {text}


def read_file_contents(file_path):
    #Reads the contents of a text file given its file path.
    with open(file_path, "r") as file:
        contents = file.read()
    return contents

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



# Directory to read PDFs
directory = '/input_and_output/PDFs'
directory_path = Path(directory)

# Path to save the CSV file
csv_file_path = '/input_and_output/extracted_data.csv'

# fieldnames (aka Column headers for the CSV output)
# the list of fieldnames should match the list of fieldnames in the assistant_message JSON example
fieldnames = read_file_contents('/input_and_output/fieldnames.txt').splitlines()


# Open CSV file for writing
with open(csv_file_path, 'w', newline='') as csv_file:
    # Define CSV writer
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write the header
    csv_writer.writeheader()

    for pdf_path in directory_path.glob('*.pdf'):
        print(pdf_path.name, flush=True)
        # Import all text (and nothing but text) from the PDF
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()

        # Create Azure client
        client = AzureOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],  
            api_version=os.environ['API_VERSION'],
            azure_endpoint=os.environ['OPENAI_API_BASE'],
            organization=os.environ['OPENAI_ORGANIZATION']
        )   

        # Create Query
        messages = [
                {"role": "system", "content": system_message()},
                {"role": "assistant", "content": assistant_message()},
                {"role": "user", "content": user_message(text=pdf_text)}
            ]


        # Make the call to the UMGPT Toolkit Azure API
        response = client.chat.completions.create(
            model=os.environ['MODEL'],
            messages=messages,
            temperature=0.0,
            response_format={ "type": "json_object" }
        )

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
                # Use the new helper function to ensure the value is a plain string
                row_data[fieldname] = normalize_value(value)

            csv_writer.writerow(row_data)
            # ---------------------------------

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

        # Wait X seconds between requests our of respect for the API service and to avoid throttling
        # Maybe not necessary if each API call takes more than a few seconds, 
        # which depends in part on how large the files are: 
        time.sleep(1)

print(f"Data successfully written to {csv_file_path}")

# Print the CSV contents
#print_csv_contents(csv_file_path)

# Function to read and print CSV contents
def print_csv_contents(file_path):
    with open(file_path, 'r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            print(row)

