"""main function module to extract data from PDFs."""


def extract_po_data(pdf_path: str) -> None:
    """
    Extract all relevant data from a PDF:
      - Table fields
      - Non-table fields
    Returns a unified list of dictionaries ready to insert into Excel.
    """
