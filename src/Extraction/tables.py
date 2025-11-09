"""Extraction modue for table data from PDFs."""
from dataclasses import Field
from math import fabs
import camelot
import tabula
import re
import camelot.core
import fitz
from camelot.io import read_pdf
from utils.field import find_field_location, find_location_of_all_fields
import pandas as pd
import numpy as np
from src.Preprocessing.preprocess import get_data_by_pattern, split_by_pack_and_column, split_combined_columns_df
from src.View.view import display_nested_dict
from utils.dict import combine_dicts
from utils.pdf import get_pdf_directory, get_pdf_page_dimensions, get_pdf_total_pages
from src.View.view import display_tables


PO_HORIZONTAL_FIELDS = [
    "Delivery Addr./Distrib. Center",
    "PO-Date",
    "Delivery Date",
    "Supplier No.",
    "Incoterms",
    "Mode of Transp",
    "Label",
    "Customer PO Number",
    "Bank Name",
    "SWIFT Code",
    "Account Name",
    "Account No",
    "Type of Freight",
    "Types of Hangers",
    "Total Number of Pieces",
    "Terms of Payment",
    "Total Amount",
    "Page"
]
# EAN MAP to color values
PO_VERTICAL_FIELDS = [
    "Required  Sustainable Certification",
    "EAN",
    "ARTICLE NO."
]

PO_BLOCK_FIELDS = ["Global Management Services Ltd."]

COLOR_HORIZONTAL_FIELDS = [
    "BRAND",
    "SAP Order No.",
    "Order Cycle",
    "Country of Origin",
    "Article",
    "Product",
    "Season",
    "Article No.",
    "Composition",
    "Year",
    "Supplier No.",
    "Color Desc.",
    "Season"

]
COLOR_VERTICAL_FIELDS = [
    "Color Desc.",
]

# FIELDS_TO_EXTRACT = HORIZONTAL_FIELDS + VERTICAL_FIELDS + HEADER_FIELDS

FINAL_EXTRACTED_VALUE = []


def extract_tables_from_page(pdf_path: str, page_num: int,
                             use_camelot=True, use_tabula=False) -> camelot.core.TableList | list[pd.DataFrame]:
    """
    Extract all tables from a single page.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        use_camelot: Whether to use Camelot for extraction
        use_tabula: Whether to use Tabula for extraction

    Returns:
        Camelot TableList or list of pandas DataFrames
    """
    x0, y0, x1, y1 = get_pdf_page_dimensions(pdf_path, page_num)

    print(
        f"📄 Reading page {page_num + 1} with dimensions: ({x0}, {y0}, {x1}, {y1})")

    try:

        if use_camelot:
            tables = read_pdf(
                pdf_path,
                flavor="stream",
                pages=str(page_num + 1),
                table_areas=[f"{x0},{y1},{x1},{y0}"],


            )

        elif use_tabula and not use_camelot:
            tables = tabula.read_pdf(
                pdf_path, pages=page_num + 1, multiple_tables=True)

        if not tables:
            print(f"❌ No tables found on page {page_num + 1}")

        return tables

    except Exception as e:
        print(f"Error in extraction ")


def extract_page_data(pdf_path: str, page_num: int) -> camelot.core.TableList | list[pd.DataFrame]:
    """
    Extract all pack data from a single page.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)

    Returns:
        Dictionary: {"Pack 1": {"group": df, "unique": df}}
    """
    tables = extract_tables_from_page(pdf_path, page_num, use_tabula=False)

    # page_packs = {}
    # for table in tables:
    #     # packs = process_table_to_packs(table, page_num)
    #     page_packs.update(table)
    return tables


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


def extract_table_data(pdf_path: str, page: str = "all") -> dict[int, dict[str, dict[str, pd.DataFrame]]]:
    """
    Extract data from all specified pages.

    Args:
        pdf_path: Path to PDF file
        page: "all" or specific page number as string

    Returns:
        Dictionary:}
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
    Identify pairs of misaligned columns using stack logic.
    Handles ANY order of header/data columns.

    Returns:
        List of tuples ALWAYS in format: (header_col_index, data_col_index)
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if not (0 <= header_row < len(df)):
        raise IndexError(
            f"header_row {header_row} out of bounds for {len(df)} rows.")
    if not (0 <= start_row < len(df)):
        raise IndexError(
            f"start_row {start_row} out of bounds for {len(df)} rows.")

    stack = []  # Store tuples: ('header'|'data', col_idx)
    result_pairs = []

    for col_idx in range(df.shape[1]):
        header = df.iloc[header_row, col_idx]

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

        # Determine column type
        if header_empty:  # DATA column
            column_type = 'data'
        elif rest_empty:  # HEADER column
            column_type = 'header'
        else:
            # Normal column - skip
            continue

        # Stack pairing logic
        if stack:
            stack_type, stack_col = stack[-1]

            # Only pair if types are different (header + data)
            if stack_type != column_type:
                stack.pop()

                # Always return in (header, data) format
                if column_type == 'header':
                    # current is header, stack is data
                    result_pairs.append((col_idx, stack_col))
                else:
                    # stack is header, current is data
                    result_pairs.append((stack_col, col_idx))
            else:
                # Same type - just push current to stack
                stack.append((column_type, col_idx))
        else:
            # Stack empty - push current
            stack.append((column_type, col_idx))

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
        values = df.iloc[start_row:, data_col]
        clean_values = values.dropna(
        ).loc[lambda x: x != ''].astype(str).tolist()
        extracted[header_name] = clean_values
        FINAL_EXTRACTED_VALUE.append(header_name)

    return extracted


def extract_all_values_from_field(
    df: pd.DataFrame,
    locations: dict[str, tuple[int, int]],
) -> dict[str, list[str] | str]:
    """
    Extracts all non-empty values under a specific field name in the DataFrame.

    Args:
        df: The DataFrame to extract from
        field_name: The field name to search for (if row/col not provided)
        row: Starting row index (default: 0)
        col: Column index (if None, will search for field_name)
        vertical: If True, extract column-wise; if False, extract row-wise
    Returns:
        List of extracted non-empty values (excluding the field name itself)
    Raises:
        ValueError: If field not found or indices out of bounds
    """
    extracted_values = {}

    for field_name, (row, col) in locations.items():
        vertical = field_name in VERTICAL_FIELDS
        values = extract_value_from_field(
            df, field_name=field_name, row=row, col=col, vertical=vertical)
        extracted_values[field_name] = values
        FINAL_EXTRACTED_VALUE.append(field_name)
        extracted_values[field_name] = values

    return extracted_values


def extract_value_from_field(
    df: pd.DataFrame,
    row: int,
    col: int,
    vertical: bool = False,

) -> str:
    """
    Extracts the immediate value that comes after ':' for a given field.
    If multi_line=True, it concatenates next `step_size` rows (vertical fields only)
    into a single string.
    """
    try:
        total_rows, total_cols = df.shape

        # Validate field position
        if row >= total_rows or col >= total_cols:
            return ""

        final_value = ""

        while row < total_rows:
            if not vertical:
                row_values = df.iloc[row, col:].dropna().astype(str)
            else:
                row_values = df.iloc[row+1:, col].dropna().astype(str)

            row_values = [v.strip()
                          for v in row_values if v.strip() and v.lower() != "nan"]

            # Extract after ':' if present, else full string
            processed_pieces = []
            for cell in row_values:
                match = re.search(r":\s*(.*)", cell)
                piece = match.group(1).strip() if match else cell
                # Remove inner whitespace of each word but keep words separated
                piece = " ".join(w.replace(" ", "") for w in piece.split())
                processed_pieces.append(piece)

            # Join columns of this row with single space
            row_string = " ".join(processed_pieces)

            # Concatenate rows directly (no extra space added between rows)
            final_value += row_string

            # Check next row's first column
            if row + 1 < len(df):
                next_first_col = df.iat[row + 1, 0]
                if pd.isna(next_first_col):
                    row += 1
                    continue
                else:
                    break
            row += 1

        return final_value

    except Exception as e:
        print(
            f"❌ Error extracting row {row}, col {col}: {e}")
        raise


def create_extraction_result(DATA_VALS: dict[int, dict[str, dict[str, pd.DataFrame]]]) -> dict[str, dict[str, list[str]]]:
    """
    Create final extraction result from DataFrame.
    Each pack is a separate object with its own fields.

    Args:
        DATA: {page_num: {"Pack 1": {"horizontal_fields": df, "vertical_fields": df}}}

    Returns:
        Dictionary: {"Pack 1": {"field_name": [
            values]}, "Pack 2": {"field_name": [values]}}
    """
    result = {}

    for page_num, packs in DATA_VALS.items():
        print(f"\n📄 Page {page_num}: Found {len(packs)} packs")

        for pack_name, fields in packs.items():
            FINAL_EXTRACTED_VALUE.clear()
            print(f"\n{'─' * 80}")
            print(f"📦 Processing {pack_name}")
            print('─' * 80)

            # Create separate result object for this pack
            pack_result = {}

            horizontal_df = fields.get('horizontal_fields')
            vertical_df = fields.get('vertical_fields')

            if horizontal_df is None or vertical_df is None:
                print(f"⚠️  Skipping {pack_name} due to missing DataFrames")
                continue

            # Find locations (no global tracking needed)
            misaligned_vertical_locations = find_misaligned_column_pairs(
                vertical_df)
            horizontal_locations = find_location_of_all_fields(
                horizontal_df, HORIZONTAL_FIELDS)
            vertical_locations = find_location_of_all_fields(
                vertical_df, VERTICAL_FIELDS)

            # Extract values
            misaligned_vertical_values = extract_misaligned_columns_values(
                vertical_df, misaligned_vertical_locations)
            horizontal_values = extract_all_values_from_field(
                horizontal_df, horizontal_locations)
            vertical_values = extract_all_values_from_field(
                vertical_df, vertical_locations)

            # Combine into pack result
            pack_result = combine_dicts(horizontal_values, vertical_values,
                                        misaligned_vertical_values)

            # Store this pack's results
            result[pack_name] = pack_result

            print(f"✅ {pack_name}: extracted {len(pack_result)} fields")

    return result


def get_header_from_df(pdf_path, page_num=0) -> pd.DataFrame | None:
    """
    Extract header fields from the DataFrame.

    Args:
        df: DataFrame to extract from
    Returns:
        Dictionary of header fields and their values
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
    for table in tables:
        df = table.df
        header = get_data_by_pattern(df, 'Pack 1')
        return header


def extract_header_values(df: pd.DataFrame, coordinates: dict[str, tuple[int, int]]) -> dict[str, str]:
    """
    Extract header field values by scanning rightwards from each field location
    until a non-empty value is found.

    Args:
        df (pd.DataFrame): DataFrame containing the table.
        coordinates (dict[str, tuple[int, int]]): Mapping of field names to (row, col) positions.

    Returns:
        dict[str, str]: Field name -> extracted value.
    """
    result = {}

    for field, (row, col) in coordinates.items():
        total_cols = df.shape[1]
        value = None

        # Start scanning one column to the right
        for c in range(col + 1, total_cols):
            candidate = df.iat[row, c]
            if not (pd.isna(candidate) or str(candidate).strip() == ""):
                value = str(candidate).strip()
                break

        # If nothing found, leave empty string
        result[field] = value if value is not None else ""

    return result


def get_pdf_json(PDF_PATH: str) -> tuple[dict[str, str], dict[str, dict[str, list[str]]]]:
    DATA_PAGE = extract_table_data(PDF_PATH, page='all')
    header = get_header_from_df(PDF_PATH)
    header_coords = find_location_of_all_fields(header, HEADER_FIELDS)
    header_values = extract_header_values(header, header_coords)

    non_header_values = create_extraction_result(DATA_PAGE)
    return header_values, non_header_values


def find_page_with_field(pages:  dict[int, dict[str, dict[str, pd.DataFrame]]], fields: list[str], page_only=False) -> dict[int, bool] | int:
    fields_page_directory = {}
    for page, tables in pages.items():

        for table in tables:
            df = table.df
            processed_df = split_combined_columns_df(df, '\n')
            for field in fields:
                exists = find_field_location(processed_df, field)
                if exists:
                    if not page_only:
                        fields_page_directory[page-1] = True
                        break
                    else:
                        return page

    return fields_page_directory


def table_in_multiple_pages(start_page: int, end_page: int, pdf_path: str) -> pd.DataFrame:
    import pandas as pd
    import camelot

    combined_df = pd.DataFrame()  # Initialize empty DF to stack sections

    for page_num in range(start_page, end_page + 1):
        x0, y0, x1, y1 = get_pdf_page_dimensions(pdf_path, page_num - 1)
        tables = read_pdf(
            pdf_path,
            flavor="stream",
            pages=str(page_num),
            table_areas=[f"{x0},{y1},{x1},{y0}"],
        )

        for table in tables:
            df = table.df
            processed_df = split_combined_columns_df(df, "\n")

            # Extract the first section (EAN → Registered Office) only on first page
            if page_num == start_page:
                section_df = get_data_by_pattern(
                    processed_df, "EAN", "Registered Office", mode='between'
                )
                combined_df = section_df.copy()  # Initialize combined_df with first section
            else:
                # For other pages, extract rows between 'Page' and 'Bank Name'
                r1, c1 = find_field_location(processed_df, "Page")
                r2, c2 = find_field_location(processed_df, "Bank Name")
                section_df = processed_df.iloc[r1 + 1:r2, c1 - 1:]

                # Stack this section below the previous combined_df
                combined_df = pd.concat(
                    [combined_df, section_df], axis=0, ignore_index=True)

    return combined_df


if __name__ == '__main__':
    PDF_PATH = str(get_pdf_directory(
        'data', 'PO10034143-V1_GHK-M000040894_Redacted.pdf'))
    # hv, nhv = get_pdf_json(PDF_PATH)
    # last_page = get_pdf_total_pages(PDF_PATH)
    # pages = extract_table_data(PDF_PATH, 'all')

    # pages_to_avoid = find_page_with_field(
    #     pages, fields=[
    #         "Label",
    #         "Customer PO Number",
    #         "Type of Freight",
    #         "Types of Hangers",
    #         "Term of Payment",
    #         "Total Number of Pieces",
    #         "Total Amount",
    #         "Required Sustainable Certification",
    #         "EAN"
    #     ]
    # )
    # ean_page = find_page_with_field(
    #     pages, fields=[
    #         "EAN"
    #     ], page_only=True
    # )
    # print(ean_page)
    # print(last_page)
    table_ean = table_in_multiple_pages(48, 49, pdf_path=PDF_PATH)
    rows = table_ean.shape[0]

    for r in range(rows):
        info = table_ean.iloc[r, :].astype(str).to_string()
        print(info)
        print('\n\n')

    # for page, tables in pages.items():

    #     # print(f"Page {page}")
    #     print('\n\n')
    #     for table in tables:
    #         df = table.df
    #         processed_df = split_combined_columns_df(df, '\n')
    # split_df=get_data_by_pattern(processed_df,'Page',mode='after')

    # footer = get_data_by_pattern(
    #     processed_df, 'Bank Name', mode='after')

    # header = get_data_by_pattern(
    #     processed_df, "PO-Date", "Page", mode="between")

    # article = get_data_by_pattern(
    #     processed_df, "ARTICLE", "DESCR.", mode='between')

    # freight = get_data_by_pattern(
    #     processed_df, "Type of Freight", " Production only Allowed at", mode='between')

    # certifcation_section = get_data_by_pattern(
    #     processed_df, "Required  Sustainable Certification", "Registered Office", mode='between')

    # ean_section = get_data_by_pattern(
    #     processed_df, "EAN", "Registered Office", mode='between')
    # sections1 = [footer, header]
    # sections2 = [article, freight, certifcation_section]

    # for section in sections:
    # if page-1 not in pages_to_avoid.keys():
    #     print(f"Page {page}")
    #     print("\n\n")

    #     for section in sections1:
    #         horizontal_locations = find_location_of_all_fields(
    #             section, PO_HORIZONTAL_FIELDS)

    #         for field, location in horizontal_locations.items():

    #             row, col = location
    #             print(field)
    #             print(extract_value_from_field(
    #                 section, row, col, vertical=False))
    #             print('\n\n')

    #         print("\n\n")
    # else:
    # print(f"Page {page}")
    # print("\n\n")

    # for section in sections2:
    #     horizontal_locations = find_location_of_all_fields(
    #         section, PO_HORIZONTAL_FIELDS)
    #     vertical_locations = find_location_of_all_fields(
    #         section, PO_VERTICAL_FIELDS)

    #     for field, location in vertical_locations.items():

    #         row, col = location
    #         print(field)
    #         print('\n\n')

    #         print(extract_value_from_field(
    #             section, row, col, vertical=True))
    #         print('\n\n')
    #     for field, location in horizontal_locations.items():

    #         row, col = location
    #         print(field)
    #         print('\n\n')

    #         print(extract_value_from_field(
    #             section, row, col, vertical=False))
    #         print('\n\n')

    #     print("\n\n")
