from camelot.io import read_pdf
from utils.pdfPath import get_pdf_path
from pathlib import Path
from src.Extraction.text import extract_text


def main():
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "data"

    for file_path in data_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".pdf":
            print(extract_text(file_path))


if __name__ == "__main__":
    main()
