import pdfplumber
import re


def extract_text(pdf_path: str):
    """Return all pages joined into one string."""
    try:
        text = []
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                print(f"❌ No pages found in {pdf_path}")
                return ""
            page_text = pdf.pages[0].extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)

    except Exception as e:
        print(f"❌ Error extracting text from {pdf_path}: {e}")
        return ""


def extractNonTableValues(text: str):
    """Extract non-table fields from PDF text."""
    try:
        non_table_fields = [
            "Order No",
            "Country",
            "Product Description",
            "Season",
            "Type of Construction",
            "No of Pieces",
            "Sales Mode",
        ]
        extracted_non_table = {}

        for field in non_table_fields:
            pattern = rf"{re.escape(field)}:\s*(.+?)(?=\n|$)"
            match = re.search(pattern, text)
            if match:
                extracted_non_table[field] = match.group(1)

        return extracted_non_table

    except Exception as e:
        print(f"❌ Error extracting non-table fields: {e}")
        return {}
