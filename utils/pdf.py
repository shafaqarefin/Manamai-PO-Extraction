"""Utility functions for PDF handling."""
from pathlib import Path
import fitz
import pandas as pd  # PyMuPDF


def get_pdf_directory(foldername: str, filename='') -> Path:
    """Get the path to the PDF directory or a specific file within it."""
    return Path(__file__).resolve().parents[1] / foldername/filename


def create_pdf_id(data_dir, split_char: str = '_') -> list[str]:
    """
    Create a list of PDF IDs from filenames in the specified directory.

    Args:
        data_dir (Path): Directory containing PDF files.
        split_char (str): Character to split the filename for ID extraction.    
    Returns:
    list[str]: List of PDF IDs.
    """
    pdf_ids = []
    for file_path in data_dir.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
            continue

        pdf_id = file_path.name.split(split_char)[0]
        pdf_ids.append(pdf_id)
    return pdf_ids


def get_pdf_page_dimensions(pdf_path: str, page_num: int) -> tuple:
    """
    Get dimensions of a specific PDF page.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)

    Returns:
        Tuple of (x0, y0, x1, y1)
    """
    doc = fitz.open(pdf_path)
    page_obj = doc[page_num]
    rect = page_obj.rect
    doc.close()
    return rect.x0, rect.y0, rect.x1, rect.y1


def get_pdf_total_pages(pdf_path: str, ) -> int:
    """
    Returns total number of pages of provided pdf

    Args:
        pdf_path (str): path to directory of pdf

    Returns:
        int: total pages
    """
    doc = fitz.open(pdf_path)

    return len(doc)
