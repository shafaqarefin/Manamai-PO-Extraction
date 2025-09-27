import pdfplumber
import re


def extract_text(pdf_path: str):
    """Return all pages joined into one string."""
    text = []

    with pdfplumber.open(pdf_path) as pdf:
        page_text = pdf.pages[0].extract_text()
        if page_text:

            text.append(page_text)

    # print(text)
    return "\n".join(text)


def extractNonTableValues(text: str):
    non_table_fields = ["Order No",
                        "Country",
                        "Product Description",
                        "Season",
                        "Type of Construction",
                        "No of Pieces",
                        "Sales Mode",]
    extracted_non_table = {}

    for field in non_table_fields:
        # Capture only the first non-whitespace sequence immediately after colon
        pattern = rf"{re.escape(field)}:\s*(.+?)(?=\n|$)"
        match = re.search(pattern, text)
        if match:
            extracted_non_table[field] = match.group(1)
    return extracted_non_table
