from pathlib import Path
from src.Excel.createExcel import save_excel_for_pdf
from src.Extraction.main import extract_PO_data
from utils.pdf import get_pdf_directory


def main():
    data_dir = get_pdf_directory('data')
    print("Creating Excel files...")

    for file_path in data_dir.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
            continue

        pdf_id = file_path.stem.split('_')[0]

        try:
            excel_data = extract_PO_data(str(file_path))
            if not excel_data:
                print(f"⚠️ No data extracted from {file_path.name}, skipping.")
                continue

            save_excel_for_pdf(pdf_id, excel_data)

        except Exception as e:
            print(f"❌ Failed processing {file_path.name}: {e}")


if __name__ == "__main__":
    main()
