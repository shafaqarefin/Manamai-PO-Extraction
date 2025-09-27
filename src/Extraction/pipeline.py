from src.Extraction.tables import extractTableValues
from src.Extraction.text import extract_text, extractNonTableValues


def extract_PO_data(pdf_path: str):
    """
    Extract all relevant data from a PDF:
      - Table fields
      - Non-table fields
    Returns a unified list of dictionaries ready to insert into Excel.
    """
    text = extract_text(pdf_path)
    table_objects = extractTableValues(pdf_path)  # list of dicts
    non_table_object = extractNonTableValues(text)  # dict

    # Combine each table row dict with non-table values
    excel_objects = []
    for table_object in table_objects:
        merged = {**non_table_object, **table_object}  # merge dicts
        excel_objects.append(merged)

    return excel_objects
