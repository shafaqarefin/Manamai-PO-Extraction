"""main function module to extract data from PDFs."""


from src.Extraction.tables import get_pdf_json


def extract_po_data(pdf_path: str):
    """
    Extract all relevant data from a PDF:
      - Table fields
      - Non-table fields
    Returns a unified list of dictionaries ready to insert into Excel.
    """
    header, non_header = get_pdf_json(pdf_path)

    return header, non_header
