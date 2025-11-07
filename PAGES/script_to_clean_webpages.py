import os
import re
from tqdm import tqdm

def process_html_to_txt(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: '{os.path.abspath(output_dir)}'")

    # Get a list of all HTML files in the input directory
    try:
        html_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.html')]
        if not html_files:
            print(f"Error: No HTML files found in the directory '{input_dir}'. Please check the path.")
            return
    except FileNotFoundError:
        print(f"Error: The input directory '{input_dir}' was not found.")
        return

    # Process each HTML file with a progress bar
    for filename in tqdm(html_files, desc="Cleaning HTML files"):
        html_path = os.path.join(input_dir, filename)
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)

        try:
            # --- 1. Read the raw text from the HTML file ---
            with open(html_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            if not raw_text.strip():
                print(f"\nWarning: No text could be extracted from '{filename}'. Skipping.")
                continue

            # --- 2. Process the text to remove header/footer ---
            # Define markers to find the main content
            primary_start_marker = "SCU Home"
            fallback_start_marker = "Home"
            footer_marker = "Core Curriculum Sections"

            # Find the start index using the primary marker first
            start_index = raw_text.find(primary_start_marker)

            # If the primary marker is not found, use the fallback logic
            if start_index == -1:
                start_index = raw_text.find(fallback_start_marker)

            if start_index == -1:
                print(f"\nWarning: Could not find a start marker in '{filename}'. Using start of the document.")
                start_index = 0 # Fallback to the very beginning of the document

            # Find the end index (the last time the footer marker appears)
            end_index = raw_text.rfind(footer_marker)
            if end_index == -1:
                # If the footer isn't found, take everything until the end of the text
                print(f"\nWarning: Footer marker not found in '{filename}'. The output might contain footer text.")
                end_index = len(raw_text)

            # Slice the string to get only the core content
            core_content = raw_text[start_index:end_index]

            # --- 3. Save the cleaned content to a .txt file ---
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(core_content.strip())

        except Exception as e:
            print(f"\nAn error occurred while processing '{filename}': {e}")

# --- Execution ---
# NOTE: Place your .html files in a folder named 'html_files' in the same
# directory as this script, or change the path in the variable below.
input_directory = './downloaded'
output_directory = './cleaned'

# Create the input directory if it doesn't exist, so you can add your files.
if not os.path.exists(input_directory):
    os.makedirs(input_directory)
    print(f"Created '{input_directory}' directory. Please upload your HTML files there.")

# Run the cleaning process
process_html_to_txt(input_dir=input_directory, output_dir=output_directory)
print("\nCleaning process complete.")