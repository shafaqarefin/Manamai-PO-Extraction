from pathlib import Path


def get_pdf_path(filename: str, foldername: str) -> str:
    return str(Path(__file__).resolve().parents[1] / foldername / filename)
