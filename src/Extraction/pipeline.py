from src.Extraction.tables import extractTableValues
from src.Extraction.text import extract_text, extractNonTableValues


def extract_PO_data(pdf_path: str):
    """
    Extract all relevant data from a PDF:
      - Table fields
      - Non-table fields
    Returns a unified list of dictionaries ready to insert into Excel.
    """
    try:
        text = extract_text(pdf_path)
        table_objects = extractTableValues(pdf_path)
        non_table_object = extractNonTableValues(text)

        excel_objects = []
        for table_object in table_objects:
            merged = {**non_table_object, **table_object}
            excel_objects.append(merged)

        return excel_objects

    except Exception as e:
        print(f"❌ Error extracting PO data from {pdf_path}: {e}")
        return []
