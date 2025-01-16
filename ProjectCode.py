#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
from dateutil.parser import parse
from transformers import pipeline

import subprocess


# Parent directory containing all folders
PARENT_DIR = "/Users/marcomontenegro/owl"

# Specific folders to process
FOLDERS = [
    "GWR Portfolio Submissions 20-21",
]

# Construct full paths for each folder
FOLDER_PATHS = [os.path.join(PARENT_DIR, folder) for folder in FOLDERS]

# Initialize a list to store extracted metadata
data = []
missing_files_data = []

# Define the 2020-2021 academic quarter date ranges
QUARTERS = {
    "Fall": ("09-17", "12-11"),
    "Winter": ("01-04", "03-19"),
    "Spring": ("03-29", "06-11"),
    "Summer": ("06-12", "09-16"),
}

# Initialize Hugging Face pipelines
metacognition_analyzer = pipeline("text-classification", model="textattack/bert-base-uncased-yelp-polarity")
trait_analyzer = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def extract_docx_text(filepath):
    """Extract text from a .docx file."""
    try:
        doc = Document(filepath)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def extract_pdf_text(filepath):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def extract_doc_text(filepath):
    """Extract text from .doc files using unoconv to convert to .docx."""
    try:
        converted_path = filepath.replace(".doc", ".docx")
        subprocess.run(["unoconv", "-f", "docx", filepath], check=True)
        doc = Document(converted_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        os.remove(converted_path)
        return text
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return ""

def extract_submission_type(filename):
    """Extract submission type (Reflection, Paper 1, Paper 2) from filename."""
    try:
        if "Reflection" in filename:
            return "Reflection"
        elif "Paper 1" in filename or "Paper1" in filename:
            return "Paper 1"
        elif "Paper 2" in filename or "Paper2" in filename:
            return "Paper 2"
        else:
            return "Unknown"
    except Exception as e:
        print(f"Error extracting submission type from filename: {e}")
        return "Unknown"

def parse_submission_date(date_str):
    """Parse a date string and return the date."""
    try:
        return parse(date_str, fuzzy=True).date()
    except Exception as e:
        print(f"Error parsing date: {date_str} -> {e}")
        return None

def assign_quarter(date):
    """Assign a quarter based on the submission date."""
    if date is None:
        return "Unknown"
    
    year = date.year
    if year not in {2020, 2021}:
        year = None  # Ignore the year if it's outside the range

    month_day = date.strftime("%m-%d")
    for quarter, (start, end) in QUARTERS.items():
        if start <= month_day <= end:
            return f"{quarter} {year}" if year else quarter
    return "Unknown"

def process_submission_date_and_quarter(text):
    """Extract and process submission date and assign a quarter."""
    lines = text.splitlines()
    for line in lines:
        date = parse_submission_date(line)
        if date:
            quarter = assign_quarter(date)
            if quarter != "Unknown":
                return date, quarter  # Includes quarter, with year if applicable
    return "Unknown", "Unknown"

def analyze_metacognition(text):
    """Analyze text for reflective and metacognitive content."""
    try:
        result = metacognition_analyzer(text[:512])  # Limit to 512 tokens
        return result[0]["label"], result[0]["score"]
    except Exception as e:
        print(f"Error in metacognition analysis: {e}")
        return "Error", 0.0

def analyze_traits(text):
    """Analyze text for the five traits."""
    traits = ["Purpose", "Synthesis", "Support", "Style", "Mechanics"]
    trait_scores = {trait: 0.0 for trait in traits}
    try:
        for trait in traits:
            result = trait_analyzer(f"{trait}: {text[:512]}")  # Customize for each trait
            trait_scores[trait] = result[0]["score"]
    except Exception as e:
        print(f"Error analyzing traits: {e}")
    return trait_scores

def categorize_paper(text, submission_type):
    """Categorize the type of paper based on its content and submission type."""
    if submission_type == "Reflection":
        return "Reflection"

    text_lower = text.lower()

    categories = {
        "Lab Report": ["experiment", "results", "methodology", "scientific study"],
        "Essay": ["thesis", "critical analysis", "persuasive", "compare and contrast"],
        "Research Paper": ["data", "research", "statistical analysis", "field study"],
        "Creative Writing": ["creative writing", "poem", "short story", "narrative"],
        "Case Study": ["case study", "business case", "scenario analysis"],
        "Presentation": ["presentation", "slides", "powerpoint", "visuals", "graphics"],
        "Report": ["report", "summary", "overview", "findings"]
    }

    for category, keywords in categories.items():
        if any(keyword in text_lower for keyword in keywords):
            return category

    return "Other"

def extract_name_from_filename(filename):
    """Extract name from the filename."""
    try:
        match = re.search(r"_(\D+?)_([A-Z])_", filename)
        if match:
            last_name = match.group(1).capitalize()
            first_initial = match.group(2).capitalize()
            return f"{first_initial}. {last_name}"
        return "Unknown"
    except Exception as e:
        print(f"Error extracting name from filename: {e}")
        return "Unknown"

def parse_metadata_from_text(text, filename):
    """Extract metadata and classify paper."""
    try:
        name = extract_name_from_filename(filename)
        submission_type = extract_submission_type(filename)
        submission_date, quarter = process_submission_date_and_quarter(text)
        paper_type = categorize_paper(text, submission_type)
        reflective_label, reflective_score = analyze_metacognition(text)
        trait_scores = analyze_traits(text)

        return {
            "name": name,
            "paper_type": paper_type,
            "submission_type": submission_type,
            "submission_date": submission_date.strftime("%Y-%m-%d") if submission_date != "Unknown" else "Unknown",
            "quarter": quarter,
            "word_count": len(text.split()),
            "metacognition_label": reflective_label,
            "metacognition_score": reflective_score,
            **trait_scores
        }
    except Exception as e:
        print(f"Error parsing metadata for {filename}: {e}")
        return {
            "name": "Unknown",
            "paper_type": "Unknown",
            "submission_type": "Unknown",
            "submission_date": "Unknown",
            "quarter": "Unknown",
            "word_count": 0,
            "metacognition_label": "Error",
            "metacognition_score": 0.0,
            "Purpose": 0.0,
            "Synthesis": 0.0,
            "Support": 0.0,
            "Style": 0.0,
            "Mechanics": 0.0,
        }

def process_files(folder_path):
    """Process files in the specified folder."""
    for root, _, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            if file.endswith(".docx"):
                text = extract_docx_text(filepath)
            elif file.endswith(".pdf"):
                text = extract_pdf_text(filepath)
            elif file.endswith(".doc"):
                text = extract_doc_text(filepath)
            else:
                print(f"Unsupported file type: {file}")
                missing_files_data.append({
                    "filename": file,
                    "filepath": filepath,
                    "name": "Unknown",
                    "paper_type": "Unknown",
                    "submission_type": "Unknown",
                    "submission_date": "Unknown",
                    "quarter": "Unknown",
                    "word_count": 0,
                    "metacognition_label": "Unknown",
                    "metacognition_score": 0.0,
                    "Purpose": 0.0,
                    "Synthesis": 0.0,
                    "Support": 0.0,
                    "Style": 0.0,
                    "Mechanics": 0.0,
                })
                continue

            metadata = parse_metadata_from_text(text, file)
            metadata["filename"] = file
            metadata["filepath"] = filepath
            data.append(metadata)

# Process files in all specified folders
if __name__ == "__main__":
    for folder_path in FOLDER_PATHS:
        process_files(folder_path)

# Save metadata to a DataFrame
df = pd.DataFrame(data)

# Separate DataFrames for each type
reflection_df = df[df["submission_type"] == "Reflection"]
paper1_df = df[df["submission_type"] == "Paper 1"]
paper2_df = df[df["submission_type"] == "Paper 2"]
missing_files_df = pd.DataFrame(missing_files_data)

# Export DataFrames
reflection_df.to_csv("reflection_submissions.csv", index=False)
paper1_df.to_csv("paper1_submissions.csv", index=False)
paper2_df.to_csv("paper2_submissions.csv", index=False)
missing_files_df.to_csv("missing_files.csv", index=False)

print("Submissions categorized and saved to separate files: reflection, paper1, paper2, and missing files.")
