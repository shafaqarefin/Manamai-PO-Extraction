from pathlib import Path


def get_pdf_directory(foldername: str) -> Path:
    return Path(__file__).resolve().parents[1] / foldername
