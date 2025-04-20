
import os
import openai
import json
import csv

# Configuration
# API_KEY = os.environ['OPENAI_API_KEY']  # Set your API key in environment variables
os.environ['OPENAI_API_KEY']='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'

client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def read_csv(filepath):
    """Reads CSV and returns list of rows (as lists of strings)."""
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
    return rows

def write_csv(filepath, rows):
    """Writes rows (list of lists) to CSV."""
    with open(filepath, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

def build_prompt(csv_rows, entity_type):
    """
    Builds a prompt that instructs OpenAI to filter out rows which are not valid character or location names.
    The CSV format is assumed to be: name,normalized_name,mentions.
    """
    csv_content = "\n".join([",".join(row) for row in csv_rows])
    prompt = (
        f"You are an expert in film script analysis. You are given a CSV file that contains extracted {entity_type} data "
        f"from a film script draft (a machine extracted that). Each row has three columns: the raw {entity_type} name, the normalized {entity_type} name, and the number of mentions. "
        f"Some rows are false positives (for example: 'maintenance shaft', 'east corridor', 'hidden panel', etc.) that are not valid {entity_type}s. \n\n"
        f"Filter the CSV based on your expertise by removing any rows that do not represent valid {entity_type} names, or those that are not relevant to the film script, or those that are not mentioned enough (or anything else random, or irrelevant, or not important, or etc...), and remove redundancies while summing their mentions, but keep {entity_type} names (personal names) even if they are mentioned few times, keeping only the main 4 or 5 {entity_type}s (or less if you see fit). \n\n"
        "Return the filtered CSV in exactly the same format (with the same columns and order) as the input. Return only CSV content!\n\n"
        "Here is the CSV data:\n"
        f"{csv_content}\n\n"
        "Filtered CSV:"
    )
    return prompt

def call_openai_api(prompt):
    """Calls the OpenAI API with the provided prompt and returns the response."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # You can change this to gpt-4 or other available models
        messages=[
            {"role": "system", "content": "You are a helpful assistant that processes CSV data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    response_text = response.choices[0].message.content
    # remove ```csv
    response_text = response_text.replace("```csv", "")
    # remove ```
    response_text = response_text.replace("```", "")
    # remove leading newlines   
    response_text = response_text.lstrip("\n")
    return response_text

# Input and output CSV file paths for characters
INPUT_CSV = "characters.csv"
OUTPUT_CSV = "filtered_characters5.csv"

# Read the CSV containing the extracted entities
csv_rows = read_csv(INPUT_CSV)

# Build the prompt that asks OpenAI to filter out invalid entities
prompt = build_prompt(csv_rows, "character")
print("Sending prompt to OpenAI API...")

filtered_csv_text = call_openai_api(prompt)

# Convert the text back into rows
filtered_rows = list(csv.reader(filtered_csv_text.splitlines()))

# Save the filtered CSV
write_csv(OUTPUT_CSV, filtered_rows)
print(f"Filtered CSV saved to {OUTPUT_CSV}")

# Input and output CSV file paths for locations
INPUT_CSV = "locations.csv"
OUTPUT_CSV = "filtered_locations5.csv"

# Read the CSV containing the extracted entities
csv_rows = read_csv(INPUT_CSV)

# Build the prompt that asks OpenAI to filter out invalid entities
prompt = build_prompt(csv_rows, "location")
print("Sending prompt to OpenAI API...")

filtered_csv_text = call_openai_api(prompt)

# Convert the text back into rows
filtered_rows = list(csv.reader(filtered_csv_text.splitlines()))

# Save the filtered CSV
write_csv(OUTPUT_CSV, filtered_rows)
print(f"Filtered CSV saved to {OUTPUT_CSV}")