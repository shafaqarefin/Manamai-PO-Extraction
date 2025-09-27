from pathlib import Path
from src.Excel.createExcel import save_excel_for_pdf
from src.Extraction.pipeline import extract_PO_data


def main():
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "data"

    for file_path in data_dir.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
            continue

        pdf_id = file_path.name.split('_')[0]
        excel_data = extract_PO_data(str(file_path))
        save_excel_for_pdf(pdf_id, excel_data)


if __name__ == "__main__":
    main()
