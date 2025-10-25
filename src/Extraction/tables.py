"""Extraction modue for table data from PDFs."""
import camelot
import camelot.core
import fitz
from camelot.io import read_pdf
import pandas as pd
import numpy as np
from src.Preprocessing.preprocess import get_data_by_pattern, split_by_pack_and_column, split_combined_columns_df
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

HEADER_FIELDS = [
    "Purchase Order Number",
    "Supplier",
    "Date Placed",
    "Bulk Order Number",
    "Buyer",
    "Department",
    "Business Unit",
    "Costing Method",
    "Freight Forwarder",
    "Currency",
    "Payment Terms",
    "Vendor Style Number",
    "Style Description",
    "Origin Port",
    "Required Handover Date",
    "Total Units", "Total Packs", "Total Cost Price"
]

FIELDS_TO_EXTRACT = HORIZONTAL_FIELDS + VERTICAL_FIELDS + HEADER_FIELDS

FINAL_EXTRACTED_VALUE = []


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


def find_location_of_all_fields(df: pd.DataFrame, fields: list[str]) -> dict[str, tuple[int, int]]:
    """
    Find (row, col) coordinates of all fields in FIELDS_TO_EXTRACT within the DataFrame.
    Handles '\n' inside cells and is case-insensitive.

    Args:
        df (pd.DataFrame): The DataFrame to search.
    Returns:
        dict[str, tuple[int, int]]: Mapping of field names to their (row, col) coordinates.
    """
    results = {}
    for field in fields:
        if field not in FINAL_EXTRACTED_VALUE:
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


def process_table_to_packs(table, page_num) -> dict:
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
        packs = process_table_to_packs(table, page_num)
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


def extract_table_data(pdf_path: str, page: str = "all") -> dict[int, dict[str, dict[str, pd.DataFrame]]]:
    """
    Extract data from all specified pages.

    Args:
        pdf_path: Path to PDF file
        page: "all" or specific page number as string

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
    field_name: str = "",
    row: int = 0,
    col: int | None = None,
    vertical: bool = True
) -> list[str]:
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
    try:
        # Find field location if col not provided
        if col is None:
            if not field_name:
                raise ValueError(
                    "Either 'col' or 'field_name' must be provided")
            row, col = find_field_location(df, field_name)

        total_rows = df.shape[0]
        total_cols = df.shape[1]

        # Validate indices
        if row >= total_rows:
            raise ValueError(
                f"Row index {row} out of bounds (DataFrame has {total_rows} rows)")
        if col >= total_cols:
            raise ValueError(
                f"Column index {col} out of bounds (DataFrame has {total_cols} columns)")

        # Extract values
        if vertical:
            # Check if there are rows below the field
            if row + 1 >= total_rows:
                print(
                    f"⚠️  No rows found under field '{field_name}' at row {row}")
                return []

            # Extract column values starting from row+1 (skip the field name itself)
            clean_list = df.iloc[row + 1:, col] \
                .dropna() \
                .loc[lambda x: x != ''] \
                .astype(str) \
                .loc[lambda x: x.str.strip().str.lower() != 'nan'] \
                .tolist()
        else:
            # Check if there are columns to the right
            if col + 1 >= total_cols:
                print(
                    f"⚠️  No columns found after field '{field_name}' at column {col}")
                return []

            # Extract row values starting from col+1 (skip the field name itself)
            clean_list = df.iloc[row, col + 1:] \
                .dropna() \
                .loc[lambda x: x != ''] \
                .astype(str) \
                .loc[lambda x: x.str.strip().str.lower() != 'nan'] \
                .tolist()

        return clean_list

    except KeyError as ke:
        print(f"❌ KeyError extracting values for field '{field_name}': {ke}")
        raise
    except Exception as e:
        print(f"❌ Error extracting values for field '{field_name}': {e}")
        raise


def create_extraction_result(DATA_VALS: dict[int, dict[str, dict[str, pd.DataFrame]]]) -> dict[str, dict[str, list[str]]]:
    """
    Create final extraction result from DataFrame.
    Each pack is a separate object with its own fields.

    Args:
        DATA: {page_num: {"Pack 1": {"horizontal_fields": df, "vertical_fields": df}}}

    Returns:
        Dictionary: {"Pack 1": {"field_name": [values]}, "Pack 2": {"field_name": [values]}}
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
            pack_result = {**horizontal_values, **vertical_values, **
                           misaligned_vertical_values}

            # Store this pack's results
            result[pack_name] = pack_result

            print(f"✅ {pack_name}: extracted {len(pack_result)} fields")

    return result


def get_header_from_df(pdf_path, page_num=0) -> pd.DataFrame:
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


if __name__ == '__main__':
    PDF_PATH = str(get_pdf_directory(
        'data', 'W25BN0518_InDC_25052005_PO_709958_BO_315814_V0.pdf'))
    hv, nhv = get_pdf_json(PDF_PATH)
    print(hv)
