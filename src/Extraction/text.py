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
        raise
