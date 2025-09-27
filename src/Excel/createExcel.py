import pandas as pd
from pathlib import Path


def save_excel_for_pdf(pdf_id: str, data_list: list, output_dir: str = "output"):
    """
    Save extracted PO data for a single PDF ID as an Excel file.

    Parameters:
    - pdf_id: str, the identifier for the PDF (used as filename)
    - data_list: list of dictionaries containing extracted data
    - output_dir: folder to save Excel file (created if doesn't exist)
    """
    if not data_list:
        print(f"No data to save for PDF ID {pdf_id}.")
        return

    # Create output folder if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert data to DataFrame and save as Excel
    df = pd.DataFrame(data_list)
    file_name = f"{pdf_id}.xlsx"
    df.to_excel(output_path / file_name, index=False)
    print(f"Saved Excel for PDF ID {pdf_id} at {output_path / file_name}")
