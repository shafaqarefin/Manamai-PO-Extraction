"""Extraction modue for table data from PDFs."""
import camelot
import camelot.core
import fitz
from camelot.io import read_pdf
import pandas as pd
import numpy as np
from src.Preprocessing.preprocess import drop_empty_columns_rows, split_by_pack_and_column, split_combined_columns_df
from src.View.view import display_nested_dict
from utils.pdf import get_pdf_directory, get_pdf_page_dimensions

HORIZONTAL_FIELDS = [
    "Pre-Pack SKU Number",
    "Shipper Required",
    "Packing Method",
    "Total Packs",
    "Unit Cost",
    "Online Pack",
]

VERTICAL_FIELDS = [
    "Buyers Colour",
    "Print",
    "Size",
    "SKU",
    "Pack Ratio",
    "Units",

]

HEADER_FIELDS = []

FIELDS_TO_EXTRACT = HORIZONTAL_FIELDS + VERTICAL_FIELDS + HEADER_FIELDS

# def extract_table_rows(pdf_path: str, page: str = "all"):
# try:
#     doc = fitz.open(pdf_path)

#     # Determine which pages to process
#     pages_to_process = range(len(doc)) if str(
#         page).lower() == "all" else [int(page)]
#     result = {}
#     for p in pages_to_process:
#         page_obj = doc[p]
#         rect = page_obj.rect  # (x0, y0, x1, y1)
#         x0, y0, x1, y1 = rect

#         print(f"📄 Reading page {p + 1} with dimensions: {rect}")

#         # Extract tables using the full page coordinates
#         tables = read_pdf(
#             pdf_path,
#             flavor="stream",
#             pages=str(p + 1),  # Camelot uses 1-based page indexing
#             table_areas=[f"{x0},{y1},{x1},{y0}"],  # full page
#         )

#         if not tables:
#             print(f"❌ No tables found on page {p + 1}")
#             continue

#         # Find all packs by each page and append to this dictionary below
#         split_packs_df_by_page = {}
#         for table_num, table in enumerate(tables, start=1):
#             df = table.df

#             fixed_df = split_combined_columns_df(df, '\n')
#             individual_packs = split_by_pack_and_column(
#                 df)
#             split_packs_df_by_page[p+1] = individual_packs

#         for key, value in split_packs_df_by_page.items():
#             print('\n\n\n')

#             for k, v in value.items():
#                 print(k)
#                 print('\n')
#                 # print(drop_empty_columns_rows(v['group']))
#                 # print('\n')
#                 # print(extract_horizontal_table_fields(
#                 #     v['group'], find_field_location(v['group'])))
#                 # print('\n\n\n')
#                 # cleaned_df = drop_empty_columns_rows(v['unique'])
#                 # if (k == "Pack 1"):
#                 #     print(extract_vertical_table_fields(
#                 #         v['unique'], find_field_location(v['unique'])))
#                 print(v['unique'])
#                 print('\n\n')

#             print('\n\n\n')

# except Exception as e:
#     print(f"❌ Error extracting table rows from {pdf_path}: {e}")
#     raise


def extract_vertical_table_fields(
    df: pd.DataFrame,
    location: dict[str, tuple],
    direction: str = "right",
    step: int = 1
) -> dict[str, list]:
    """
    Extract vertical table field values from the DataFrame using location coordinates.
    If a cell is empty, searches in the specified direction with the given step until a non-empty value is found.

    Args:
        df (pd.DataFrame): The DataFrame to extract from.
        location (dict): Dictionary mapping field names to (row, col) coordinates.
        vertical_fields (list): List of vertical field names to extract.
        direction (str): Direction to search when a cell is empty. Options: 'left', 'right', 'up', 'down'.
        step (int): Step size for each move in the chosen direction.

    Returns:
        dict[str, list]: Dictionary with field names as keys and list of extracted values.
    """
    extracted = {field: [] for field in VERTICAL_FIELDS}

    for field in VERTICAL_FIELDS:
        if field not in location:
            continue

        row, col = location[field]
        total_rows = df.shape[0]
        for r in range(row + 1, total_rows):
            value = df.iat[r, col]
            if pd.isna(value) or str(value).strip() == "":
                # fallback search in the specified direction
                found = None
                search_row, search_col = r, col

                while True:
                    if direction.lower() == "left":
                        search_col -= step
                        if search_col < 0:
                            break
                    elif direction.lower() == "right":
                        search_col += step
                        if search_col >= df.shape[1]:
                            break
                    elif direction.lower() == "up":
                        search_row -= step
                        if search_row < 0:
                            break
                    elif direction.lower() == "down":
                        search_row += step
                        if search_row >= df.shape[0]:
                            break
                    else:
                        raise ValueError(f"Invalid direction: {direction}")

                    val = df.iat[search_row, search_col]
                    if not (pd.isna(val) or str(val).strip() == ""):
                        found = val
                        break

                if found:
                    extracted[field].append(str(found).strip())
                else:
                    continue  # stop if nothing found in the chosen direction
            else:
                extracted[field].append(str(value).strip())

    return extracted


def extract_horizontal_table_fields(df: pd.DataFrame, location: dict[str, tuple[int, int]],
                                    direction: str = "right", step: int = 1) -> dict[str, list]:
    """
    Extract horizontal field values from DataFrame using location dict.

    Args:
        df (pd.DataFrame): The DataFrame to extract from.
        location (dict): Dict mapping field names to (row, col) tuples.
        direction (str): "right" (default) or "left".
        step (int): Number of columns to skip per step (default 1).

    Returns:
        dict[str, list]: Field name mapped to list of extracted values.
    """
    result = {}

    for field, (row, col) in location.items():
        values = []
        max_cols = df.shape[1]
        current_col = col+1
        while current_col < max_cols and current_col >= 0:
            cell_value = df.iat[row, current_col]

            # If empty, keep moving in the direction
            while (pd.isna(cell_value) or str(cell_value).strip() == ""):
                current_col = current_col + step if direction == "right" else current_col - step
                if current_col >= max_cols or current_col < 0:
                    break
                cell_value = df.iat[row, current_col]

            if current_col >= max_cols or current_col < 0:
                break  # Stop if we went past the DataFrame

            values.append(cell_value)
            current_col = current_col + step if direction == "right" else current_col - step

        result[field] = values

    return result


def find_field_location(df: pd.DataFrame, field: str) -> tuple[int, int]:
    """
    Efficiently find (row, col) coordinates of specific fields in the DataFrame.
    Handles '\n' inside cells and is case-insensitive.
    """
    try:
        field_lower = field.strip().lower()
        # Normalize text: string, strip, lowercase
        df_str = df.astype(str).map(lambda x: x.strip().lower())

        # Split by newline into lists
        df_split = df_str.map(
            lambda x: [p.strip().lower() for p in str(x).split('\n') if p.strip()])

        # Mask: True if any part matches field name (with/without colon)
        mask = df_split.map(
            lambda parts: any(p == field_lower or p ==
                              f"{field_lower}:" for p in parts)
        )

        rows, cols = np.where(mask.to_numpy())
        return int(rows[0]), int(cols[0])

    except Exception as e:
        print(f"Error: {e}")
        raise


def find_location_of_all_fields(df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """
    Find (row, col) coordinates of all fields in FIELDS_TO_EXTRACT within the DataFrame.
    Handles '\n' inside cells and is case-insensitive.

    Args:
        df (pd.DataFrame): The DataFrame to search.
    Returns:
        dict[str, tuple[int, int]]: Mapping of field names to their (row, col) coordinates.
    """
    results = {}
    for field in FIELDS_TO_EXTRACT:
        try:
            row, col = find_field_location(df, field)
            results[field] = (row, col)

        except LookupError:
            print(f"❌ Field '{field}' not found in DataFrame.")
            continue  # Field not found; skip
    return results


def extract_tables_from_page(pdf_path: str, page_num: int) -> camelot.core.TableList:
    """
    Extract all tables from a single page.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)

    Returns:
        List of Camelot table objects
    """
    x0, y0, x1, y1 = get_pdf_page_dimensions(pdf_path, page_num)

    print(
        f"📄 Reading page {page_num + 1} with dimensions: ({x0}, {y0}, {x1}, {y1})")

    tables = read_pdf(
        pdf_path,
        flavor="stream",
        pages=str(page_num + 1),  # Camelot uses 1-based indexing
        table_areas=[f"{x0},{y1},{x1},{y0}"],
    )

    if not tables:
        print(f"❌ No tables found on page {page_num + 1}")

    return tables


def process_table_to_packs(table) -> dict:
    """
    Process a single Camelot table into pack structure.

    Args:
        table: Camelot table object

    Returns:
        Dictionary of packs: {"Pack 1": {"group": df, "unique": df}}
    """
    df = table.df
    fixed_df = split_combined_columns_df(df, '\n')
    individual_packs = split_by_pack_and_column(fixed_df)

    return individual_packs


def extract_page_data(pdf_path: str, page_num: int) -> dict:
    """
    Extract all pack data from a single page.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)

    Returns:
        Dictionary: {"Pack 1": {"group": df, "unique": df}}
    """
    tables = extract_tables_from_page(pdf_path, page_num)

    page_packs = {}
    for table in tables:
        packs = process_table_to_packs(table)
        page_packs.update(packs)

    return page_packs


def get_pages_to_process(total_pages: int, page: str) -> list:
    """
    Determine which pages to process.

    Args:
        total_pages: Total number of pages in PDF
        page: "all" or specific page number as string

    Returns:
        List of page numbers (0-indexed)
    """
    if str(page).lower() == "all":
        return list(range(total_pages))
    else:
        return [int(page)]


def extract_data(pdf_path: str, page: str = "all") -> dict[str, dict[str, pd.DataFrame]]:
    """
    Extract data from all specified pages.

    Args:
        pdf_path: Path to PDF file
        page: "all" or specific page number

    Returns:
        Dictionary: {page_num: {"Pack 1": {
            "horizontal_fields": df, "vertical_fields": df}}}
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    pages_to_process = get_pages_to_process(total_pages, page)

    all_pages_data = {}

    for page_num in pages_to_process:
        try:
            page_data = extract_page_data(pdf_path, page_num)
            if page_data:  # Only add if data was found
                all_pages_data[page_num + 1] = page_data  # Store as 1-indexed
        except KeyError as e:
            print(f"❌ Error processing page {page_num + 1}: {e}")
            raise

    return all_pages_data


def find_misaligned_column_pairs(df: pd.DataFrame, header_row: int = 0, start_row: int = 1) -> list[tuple[int, int]]:
    """
    Identify pairs of misaligned columns using stack logic:
    - Columns where header_row has a value but rest of column is empty -> push to stack
    - Columns where header_row is empty but rest of column has values -> pop from stack and pair

    Args:
        df: pandas DataFrame (generic, any column names)
        header_row: row index to treat as header (default 0)
        start_row: row index to start checking rest of the column (default 1)

    Returns:
        List of tuples (header_col_index, data_col_index)
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if not (0 <= header_row < len(df)):
        raise IndexError(
            f"header_row {header_row} out of bounds for {len(df)} rows.")
    if not (0 <= start_row < len(df)):
        raise IndexError(
            f"start_row {start_row} out of bounds for {len(df)} rows.")

    stack = []
    result_pairs = []

    for col_idx in range(df.shape[1]):
        header = df.iloc[header_row, col_idx]
        print(f"Checking column {col_idx}: header='{header}'")
        # Get all non-empty values under the column
        clean_list = df.iloc[start_row:, col_idx] \
            .dropna() \
            .loc[lambda x: x != ''] \
            .astype(str) \
            .loc[lambda x: x.str.lower() != 'nan'] \
            .tolist()

        rest_empty = len(clean_list) == 0
        header_empty = pd.isna(header) or str(
            header).strip() in ['', 'nan', '<NA>']

        print(f"header_empty={header_empty}, rest_empty={rest_empty}")
        print('\n')
        print(f"clean_list: {clean_list}")

        # Stack logic
        if header_empty or rest_empty:

            if stack:
                result_pairs.append((stack.pop(0), col_idx))
            else:
                stack.append(col_idx)
        print('\n\n\n')

    return result_pairs


def extract_misaligned_columns_values(df: pd.DataFrame, column_pairs: list[tuple[int, int]], start_row: int = 1) -> dict[str, list[str]]:
    """_summary_
    Extract values from misaligned column pairs.

    Args:
        df (pd.DataFrame): The DataFrame to extract from.
        column_pairs (list[tuple[int, int]]): List of (header_col_index, data_col_index) pairs.
        start_row (int, optional): start value from row. Defaults to 1.

    Returns:
        dict[str, list[str]]: Dictionary with header column index as key and list of extracted values.
    """
    extracted = {}

    for header_col, data_col in column_pairs:
        header_name = str(df.iat[0, header_col]).strip()
        print(header_name)
        values = df.iloc[start_row:, data_col]
        clean_values = values.dropna(
        ).loc[lambda x: x != ''].astype(str).tolist()
        extracted[header_name] = clean_values

    return extracted


def extract_values_from_field(df: pd.DataFrame, field_name: str = "", row=0, col: int | None = None) -> list[str]:
    """
    Extracts all non-empty values under a specific field name in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to extract from.
        field_name (str): The field name to search for.

    Returns:
        list[str]: List of extracted values.
    """
    try:
        if row == 0 and col is None:
            row, col = find_field_location(df, field_name)

        total_rows = df.shape[0]
        total_cols = df.shape[1]
        if (row + 1 >= total_rows):
            print(f"❌ No rows found under field '{field_name}'")
            return []
        if (col >= total_cols):
            print(
                f"❌ Column index {col} out of bounds for field '{field_name}'")
            return []

        clean_list = df.iloc[row:, col] \
            .dropna() \
            .loc[lambda x: x != ''] \
            .astype(str) \
            .tolist()
        return clean_list

    except KeyError as ke:
        print(f"❌ Error extracting values for field '{field_name}': {ke}")
        raise


if __name__ == '__main__':

    PDF_PATH = str(get_pdf_directory(
        'data', 'PO SHEET- BEST & LESS.pdf'))
    DATA_PAGE = extract_data(PDF_PATH, page='0')
    # print(DATA_PAGE)
    # display_nested_dict(DATA_PAGE_1, title="Extracted Data from Page 14")
    horizontal_pd: pd.DataFrame = DATA_PAGE[1]['Pack 1']['horizontal_fields']
    vertical_pd: pd.DataFrame = DATA_PAGE[1]['Pack 1']['vertical_fields']
    # print('\n\n')
    # print(extract_values_from_field(vertical_pd, col=2))
    # print(extract_values_from_field(vertical_pd, col=6))
    # print(extract_values_from_field(vertical_pd, col=7))
    # print(extract_values_from_field(vertical_pd, col=8))
    # print(vertical_pd)
    # print(find_field_location(vertical_pd, "Pack Ratio"))
    COORDS = find_misaligned_column_pairs(vertical_pd)
    print(COORDS)
    print(extract_misaligned_columns_values(vertical_pd, COORDS))
    print('\n\n')
