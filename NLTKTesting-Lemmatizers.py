import os
import json

def read_text_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):  # Process only .txt files
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()  # Read the file content
                    print(f"--- {filename} ---\n{content}\n")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

# Set the folder path
text_folder = r"C:\Users\maya_\Documents\GitHub\NTLKTesting\ExtractedTxt"  # Replace with your folder path

read_text_files(text_folder)