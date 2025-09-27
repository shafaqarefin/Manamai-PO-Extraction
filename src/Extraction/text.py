import pdfplumber

from utils.pdfPath import get_pdf_path


def extract_text(pdf_path):
    """Return all pages joined into one string."""
    text = []

    with pdfplumber.open(pdf_path) as pdf:
        page_text = pdf.pages[0].extract_text()
        if page_text:

            text.append(page_text)

    # print(text)
    return "\n".join(text)


# Get the entire PDF text
pdf_path = get_pdf_path('Input_Sample.pdf', 'data')
print(extract_text(pdf_path))
