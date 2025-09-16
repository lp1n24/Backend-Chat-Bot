import os
import re
import csv
from pypdf import PdfReader

# Define the path to where all the academic PDFs are stored
pdf_dir = os.path.join("data", "papers", "papers")

# Define the path where the processed text should be stored
output_dir = os.path.join("data", "processed")
output_csv = os.path.join(output_dir, "pages.csv")

# simple pattern to detect captions in case the query include reference to specific tables
caption_re = re.compile(r"^(table|figure|fig\.)\s*\d+", re.IGNORECASE)

# Function to return a list of full file paths for all PDFs inside data folder
def find_pdfs(folder):
    pdfs = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.endswith(".pdf"):
                pdfs.append(os.path.join(root, name))
    return pdfs

# Function to extract pages from a given PDF file path, then return a list of (page_number, type, text).
# Type can be "page" for full page text and it can also be "caption" for caption lines line Table or Figure
def extract_pages(pdf_path):
    page_rows = []
    try:
        reader = PdfReader(pdf_path)
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            raw = page.extract_text() or ""

            # Keep line breaks to scan for captions
            lines = []
            for line in raw.splitlines():
                line = " ".join(line.split())
                if line:
                    lines.append(line)
            
            page_text = " ".join(lines)
            page_rows.append((i + 1, "page", page_text))

            for line in lines:
                if caption_re.match(line):
                    page_rows.append((i + 1, "caption", line))
    
    except Exception as e:
        print("could not read", pdf_path, ":", e)
    return page_rows

def main():
    os.makedirs(output_dir, exist_ok=True)
    
    pdfs = find_pdfs(pdf_dir)

    rows = []
    total_pages = 0
    valid_pages = 0 # Page without text is not counted
    caption_count = 0

    for pdf in pdfs:
        page_list = extract_pages(pdf)
        filename = os.path.basename(pdf)

        seen_pages = set()
        for page_number, type, text in page_list:
            if type == "page":
                seen_pages.add(page_number)
                if text.strip():
                    valid_pages += 1
            elif type == "caption":
                caption_count += 1

            rows.append({"source": filename, "page": page_number, "type": type, "text": text})

        total_pages += len(seen_pages)

    # Write rows into CSV file
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "page", "type", "text"])
        writer.writeheader()
        writer.writerows(rows)
        
    # Short summary for easier validation
    print(f"Processed {len(pdfs)} PDFs, Total pages: {total_pages}, Valid pages: {valid_pages}, Captions: {caption_count}")

if __name__ == "__main__":
    main()