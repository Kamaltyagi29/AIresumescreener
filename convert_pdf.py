# %%
from pypdf import PdfReader
import glob
import csv
import os
import pandas as pd

def clean_text(text):
    """Cleans extracted text by removing unwanted characters."""
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = text.replace("+", " ").replace("-", " ").replace("(", " ").replace(")", " ")
    return text

def extract_resumes_to_csv(pdf_path, out_csv):
    """
    Extracts text from a single PDF resume and appends it to the CSV file.
    
    Parameters:
    pdf_path (str): Path to the PDF file.
    out_csv (str): Path to the output CSV file.
    """
    resume_name = os.path.basename(pdf_path)
    
    # Read existing CSV if it exists
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv)
    else:
        df = pd.DataFrame(columns=["ID", "Resume_name", "Resume"])
    
    # Check if the resume is already processed
    if resume_name in df["Resume_name"].values:
        print(f"Skipping {resume_name} (Already processed)")
        return

    # Extract text from the PDF
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    text = clean_text(text)

    # Assign ID (incremental based on existing data)
    new_id = df["ID"].max() + 1 if not df.empty else 0

    # Append new data
    new_row = pd.DataFrame([[new_id, resume_name, text]], columns=["ID", "Resume_name", "Resume"])
    df = pd.concat([df, new_row], ignore_index=True)

    # Save updated CSV
    df.to_csv(out_csv, index=False)
    print(f"Added {resume_name} to {out_csv}")
