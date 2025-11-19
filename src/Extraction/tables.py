"""Extraction modue for table data from PDFs."""
from calendar import c
from turtle import color
import camelot
import tabula
import re
import camelot.core
import fitz
from camelot.io import read_pdf
from utils.field import find_field_location, find_location_of_all_fields
import pandas as pd
import numpy as np
from src.Preprocessing.preprocess import find_specific_section, get_data_by_pattern, split_by_pack_and_column, split_combined_columns_df
from src.View.view import display_nested_dict
from utils.dict import combine_dicts
from utils.pdf import get_pdf_directory, get_pdf_page_dimensions, get_pdf_total_pages
from src.View.view import display_tables
from src.Excel.excel import save_excel_for_pdf


PO_HORIZONTAL_FIELDS = [
    "PO-Date",
    "Purchase Order No.",
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


MULTI_LINE_PO_HORIZONTAL_FIELDS = [
    "Delivery Addr./Distrib. Center",
    "Bank Name",
    "Account Name",
    "Customer PO Number",
    "SAP Order No.",

]


# EAN MAP to color values
PO_VERTICAL_FIELDS = [
    "Delivery Addr./Distrib. Center",
    "Required  Sustainable Certification",
    "EAN",
    "ARTICLE NO."
    "Global Management Services Ltd."
]

PO_BLOCK_FIELDS = ["Global Management Services Ltd."]

COLOR_HORIZONTAL_FIELDS = [
    "Brand",
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


def extract_tables_from_page(pdf_path: str, page_num: int, col_tol: int, row_tol: int,
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
                row_tol=row_tol,
                column_tol=col_tol

            )

        elif use_tabula and not use_camelot:
            tables = tabula.read_pdf(
                pdf_path, pages=page_num + 1, multiple_tables=True)

        if not tables:
            print(f"❌ No tables found on page {page_num + 1}")

        return tables

    except Exception as e:
        print(f"Error in extraction ")


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


def extract_table_data(pdf_path: str, page: str = "all", col_tol: int = 100, row_tol: int = 10) -> dict[int, dict[str, dict[str, pd.DataFrame]]]:
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
            page_data = extract_tables_from_page(
                pdf_path, page_num, col_tol=col_tol, row_tol=row_tol)
            if page_data:  # Only add if data was found
                all_pages_data[page_num + 1] = page_data  # Store as 1-indexed
        except KeyError as e:
            print(f"❌ Error processing page {page_num + 1}: {e}")
            raise

    return all_pages_data


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
    multiline: bool = False,
    pattern: str = r":\s*(.*)",
    is_list: bool = False,
    space_next_line=False
) -> str | list:
    """
    Extracts the immediate value that comes after ':' for a given field.
    - If multiline=True and vertical=False, concatenates multiple rows horizontally.
    - If vertical=True, extracts downward column values (ignores multiline).
    - If is_list=True, returns a list of values instead of a single string.
    """
    try:
        total_rows, total_cols = df.shape

        # Validate position
        if row >= total_rows or col >= total_cols:
            return [] if is_list else ""

        final_value = []

        # --- CASE 1: vertical extraction (ignores multiline) ---
        if vertical:
            row_values = df.iloc[row + 1:, col].dropna().astype(str)
            row_values = [v.strip()
                          for v in row_values if v.strip() and v.lower() != "nan"]

            processed_pieces = []
            for cell in row_values:
                match = re.search(pattern, cell)
                piece = match.group(1).strip() if match else cell
                processed_pieces.append(" ".join(piece.split()))

            final_value = processed_pieces

        # --- CASE 2: multiline horizontal extraction ---
        elif multiline:
            # Loop through all remaining rows
            for i in range(row, total_rows):
                row_values = df.iloc[i, col:].dropna().astype(str)
                row_values = [v.strip()
                              for v in row_values if v.strip() and v.lower() != "nan"]

                processed_pieces = []
                for cell in row_values:
                    match = re.search(pattern, cell)
                    piece = match.group(1).strip() if match else cell
                    processed_pieces.append(
                        "".join(piece.split()) if space_next_line else " ".join(piece.split()))

                final_value.extend(processed_pieces)

        # --- CASE 3: single-line horizontal extraction ---
        else:
            row_values = df.iloc[row, col:].dropna().astype(str)
            row_values = [v.strip()
                          for v in row_values if v.strip() and v.lower() != "nan"]

            processed_pieces = []
            for cell in row_values:
                match = re.search(pattern, cell)
                piece = match.group(1).strip() if match else cell
                processed_pieces.append(" ".join(piece.split()))

            final_value = processed_pieces

        # ✅ FINAL NORMALIZATION
        if is_list:
            # Always return a list
            if not isinstance(final_value, list):
                final_value = [final_value]
        else:
            # Always flatten to a single string
            if isinstance(final_value, list):
                final_value = (
                    "".join(str(v).strip()
                            for v in final_value if str(v).strip())
                    if space_next_line
                    else " ".join(str(v).strip() for v in final_value if str(v).strip())
                )

        return final_value

    except Exception as e:
        # print(f"❌ Error extracting row {row}, col {col}: {e}")
        # raise
        pass


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


def table_in_multiple_pages(start_page: int, end_page: int, pdf_path: str) -> tuple[pd.DataFrame, str]:

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
                a_r1, a_c1 = find_field_location(processed_df, "ARTICLE NO.")
                article_number = " ".join(
                    processed_df.iloc[a_r1+1:, a_c1].astype(str).to_list()).strip()

                combined_df = section_df.copy()  # Initialize combined_df with first section
            else:
                # For other pages, extract rows between 'Page' and 'Bank Name'
                r1, c1 = find_field_location(processed_df, "Page")
                r2, c2 = find_field_location(processed_df, "Bank Name")
                section_df = processed_df.iloc[r1 + 1:r2, c1 - 1:]

                # Stack this section below the previous combined_df
                combined_df = pd.concat(
                    [combined_df, section_df], axis=0, ignore_index=True)

    return combined_df, article_number


def color_code_map(TABLE_SECTION: pd.DataFrame, header_row: int = 0, start_row: int = 1, col: int = 0) -> dict:
    """
    Converts a table of colors, sizes, and quantities into a nested dictionary
    keyed by color code.

    Args:
        TABLE_SECTION: DataFrame or list of lists containing the table.
        header_row: Row index for size headers.
        start_row: Row index where data starts.
        col: Column index where the color column starts.

    Returns:
        dict: {color_code: {"color": color_name, "size": {
            size_header: value}, "total_qty": total}}
    """
    result_dict = {}
    print(TABLE_SECTION)
    # Extract size headers
    size_headers = [v.strip() for v in extract_value_from_field(
        TABLE_SECTION, row=header_row, col=col, is_list=True) if v.strip() and v.lower() != "nan"]
    size_headers = size_headers[1:]  # skip color column and total_qty column
    # Clean last header if needed
    size_headers[-1] = "".join(size_headers[-1].split(' ')[0])
    # Iterate over data rows
    for i in range(start_row, len(TABLE_SECTION)):
        row_values = extract_value_from_field(
            TABLE_SECTION, row=i, col=col, is_list=True)
        row_values = [v.strip()
                      for v in row_values if v.strip() and v.lower() != "nan"]

        if not row_values:
            continue  # skip empty rows

        color_field = row_values[0]
        total_qty = row_values[-1].replace(',', '')  # last column is total qty

        # Extract color name and code
        m = re.match(r"(.*)\s+\((\d+)\)", color_field)
        if m:
            color_name = m.group(1).strip()
            code = m.group(2)
        else:
            color_name = color_field.strip()
            code = None  # fallback if no code

        # Build sizes dictionary
        sizes = {}
        for idx, size_header in enumerate(size_headers):
            if idx + 1 < len(row_values) - 1:
                sizes[size_header] = row_values[idx + 1].replace(',', '')

        # Use color code as key
        if code:
            result_dict[code] = {
                "color": color_name,
                "size": sizes,
                "total_qty": total_qty
            }

    return result_dict


def build_ean_lookup(ean_color_list):
    lookup = {}
    for entry in ean_color_list:
        ean = str(entry["EAN"]).strip()
        code = str(entry["Color Code"]).strip()
        size = str(entry["Size"]).strip()
        lookup[(code, size)] = ean
    return lookup


def flatten_color_size_data(ean_color_list, color_data_dict):
    flattened = []
    lookup = build_ean_lookup(ean_color_list)

    for color_code, info in color_data_dict.items():
        color_name = info["color"]
        total_qty = info["total_qty"]
        size_map = info["size"]

        for size, qty in size_map.items():

            ean = lookup.get((color_code, size))

            # 🚨 No EAN found → do not warn, just skip
            if not ean:
                continue

            flattened.append({
                "EAN": ean,
                "Color Code": color_code,
                "Color": color_name,
                "Size": size,
                "Qty": qty,
                "Total Qty": total_qty
            })

    return flattened


def extract_common_po_fields(po_data: dict[int, dict[str, dict[str, pd.DataFrame]]]) -> dict[str, str]:
    common_fields_value = {}
    pages_to_avoid = find_page_with_field(
        po_data, fields=[
            "Label",
            "Customer PO Number",
            "Type of Freight",
            "Types of Hangers",
            "Term of Payment",
            "Total Number of Pieces",
            "Total Amount",
            "Required Sustainable Certification",
            "EAN"
        ]
    )
    # PO COMMON DATA
    for page, tables in po_data.items():
        for table in tables:
            df = table.df
            processed_df = split_combined_columns_df(df, '\n')

            top_section = get_data_by_pattern(
                processed_df, 'Delivery Addr./Distrib. Center', mode='before')
            BANK_NAME = find_specific_section(
                processed_df, 'Bank Name', until_field="SWIFT Code", row_wise=True, col_wise=True, includes=False)
            SWIFT_CODE = find_specific_section(
                processed_df, 'SWIFT Code', until_field="Account Name", row_wise=True, col_wise=True, includes=False)
            ACCOUNT_NAME = find_specific_section(
                processed_df, 'Account Name', until_field="Account No", row_wise=True, col_wise=True, includes=False)
            ACCOUNT_NO = find_specific_section(
                processed_df, 'Account No', row_wise=True, col_wise=True, includes=False)

            HEADER = find_specific_section(
                processed_df, from_field="PO-Date", until_field="Page", row_wise=True, col_wise=True, includes=False)
            LABEL_SECTION = find_specific_section(
                processed_df, from_field="Label", to_field="DESCR.", row_wise=True, col_wise=False, includes=False)
            freight = get_data_by_pattern(
                processed_df, "Type of Freight", " Production only Allowed at", mode='between')
            certifcation_section = get_data_by_pattern(
                processed_df, "Required  Sustainable Certification", "Registered Office", mode='between')
            DELIVERY_ADDR_SECTION = find_specific_section(
                processed_df, 'Delivery Addr./Distrib. Center', 'PO-Date', 'Page', col_wise=True, row_wise=True, includes=False)

            TOTAL_AMOUNT_SECTION = find_specific_section(
                processed_df, from_field='Total Number of Pieces', until_field='Total Amount', col_wise=False, row_wise=True, includes=True)

            FOOTER_SECTION = find_specific_section(
                processed_df, from_field='Registered Office', col_wise=False, row_wise=True, includes=True)

            sections1 = [HEADER, BANK_NAME, SWIFT_CODE, ACCOUNT_NAME,
                         ACCOUNT_NO, top_section, DELIVERY_ADDR_SECTION]
            sections2 = [LABEL_SECTION, freight,
                         certifcation_section, TOTAL_AMOUNT_SECTION]

            if page - 1 not in pages_to_avoid.keys():
                sections_to_process = sections1
                if not FOOTER_SECTION.empty and "Buying House" not in common_fields_value and "Buying House Address" not in common_fields_value:
                    common_fields_value["Buying House"] = FOOTER_SECTION.iloc[0, 0].strip(
                    )
                    common_fields_value["Buying House Address"] = " ".join(
                        FOOTER_SECTION.iloc[1:, 0].dropna().astype(str).map(str.strip).to_list()).strip()
            else:
                sections_to_process = sections2

            for section in sections_to_process:
                # Process horizontal fields
                horizontal_locations = find_location_of_all_fields(
                    section, PO_HORIZONTAL_FIELDS)
                for field, location in horizontal_locations.items():
                    if field not in common_fields_value or not common_fields_value.get(field):
                        row, col = location
                        value = extract_value_from_field(
                            section,
                            row,
                            col,
                            vertical=False,
                            is_list=False,
                            multiline=field in MULTI_LINE_PO_HORIZONTAL_FIELDS,
                            space_next_line=field in ["Customer PO Number"]

                        )
                        if isinstance(value, str):
                            value = value.strip()
                        elif isinstance(value, list):
                            value = [v.strip() for v in value if v.strip()]

                        if field.lower() == "type of freight":
                            value = value.split(" ")[0] if value else value
                        if field.lower() == "po-date":
                            value = value.split(" ")[-1] if value else value
                        if field.lower() == "terms of payment":
                            value = " ".join(value.strip().split(" ")[
                                :-1]) if value else value

                        common_fields_value[field] = value

                # Process vertical fields
                vertical_locations = find_location_of_all_fields(
                    section, PO_VERTICAL_FIELDS)
                for field, location in vertical_locations.items():
                    if field not in common_fields_value or not common_fields_value.get(field):
                        row, col = location
                        value = extract_value_from_field(
                            section, row, col, vertical=True)
                        if isinstance(value, str):
                            value = value.strip()
                        elif isinstance(value, list):
                            value = [v.strip() for v in value if v.strip()]
                        common_fields_value[field] = value

    return common_fields_value


def extract_color_size(color_size_data):
    """
    Extract color size information and common color-related values from table data.

    Parameters:
        color_size_data (dict): A mapping of page -> list of table objects (each having a .df DataFrame)

    Returns:
        tuple: (color_size_common_values, color_size_info)
            - color_size_common_values: dict of extracted horizontal fields
            - color_size_info: dict or list from color_code_map()
    """
    color_size_common_values = {}

    for page, tables in color_size_data.items():
        for table in tables:
            df = table.df
            processed_df = split_combined_columns_df(df, '\n')

            # --- Identify key header sections ---
            LEFT_SECTION_HEADER_TOP = find_specific_section(
                processed_df, 'Brand', 'Article', "Order Cycle",
                col_wise=True, row_wise=True, includes=False
            )
            LEFT_SECTION_HEADER_BOTTOM = find_specific_section(
                processed_df, 'Order Cycle', 'Season', "Factory",
                col_wise=True, row_wise=True, includes=False
            )
            MIDDLE_SECTION_HEADER = find_specific_section(
                processed_df, from_field='Article', to_field='Article No.', until_field="Season",
                col_wise=True, row_wise=True, includes=False
            )
            LAST_SECTION_HEADER = find_specific_section(
                processed_df, 'Article No.', 'Supplier No.',
                col_wise=False, row_wise=True, includes=True
            )
            TABLE_SECTION = find_specific_section(
                processed_df, 'Color Desc.',
                col_wise=False, row_wise=True, includes=False
            )
            SEASON = find_specific_section(
                processed_df, from_field='Season', to_field='Year',
                until_field='Supplier', col_wise=True, row_wise=True, includes=False)

            sections_to_process = [
                LEFT_SECTION_HEADER_BOTTOM,
                LEFT_SECTION_HEADER_TOP,
                MIDDLE_SECTION_HEADER,
                LAST_SECTION_HEADER,
                SEASON
            ]

            # --- Extract color-size table info ---
            row, col = find_field_location(TABLE_SECTION, "Color Desc.")
            color_size_info = color_code_map(
                TABLE_SECTION, header_row=row, start_row=row + 1, col=col
            )

            # --- Extract other color-related fields ---
            for section in sections_to_process:
                horizontal_locations = find_location_of_all_fields(
                    section, COLOR_HORIZONTAL_FIELDS
                )

                for field, location in horizontal_locations.items():
                    if field not in color_size_common_values:
                        row, col = location
                        pattern = f"{field}\s*\s*(.*)"
                        value = extract_value_from_field(
                            section,
                            row,
                            col,
                            vertical=False,
                            multiline=field in MULTI_LINE_PO_HORIZONTAL_FIELDS,
                            pattern=pattern
                        )
                        if field.lower() == "season":
                            value = value.split(" ")[0] if value else value
                        color_size_common_values[field] = value

    return color_size_common_values, color_size_info


if __name__ == '__main__':
    PO_PDF_PATH = str(get_pdf_directory(
        'data', subfolder='test2', filename='PO10034143-V1_GHK-M000040894_Redacted.pdf'))
    # hv, nhv = get_pdf_json(PDF_PATH)
    last_page = get_pdf_total_pages(PO_PDF_PATH)

    po_data = extract_table_data(PO_PDF_PATH, 'all')
    print(po_data)
    ean_page = find_page_with_field(
        po_data, fields=[
            "EAN"
        ], page_only=True
    )

    table_ean, article_no = table_in_multiple_pages(
        ean_page, last_page, pdf_path=PO_PDF_PATH)

    rows = table_ean.shape[0]
    temporary_dict = {}
    list_of_ean_objects = []
    print(table_ean)
    for r in range(1, rows):
        info = table_ean.iloc[r, :].astype(str).to_list()
        val = info[1].split(" ", 1)
        EAN = info[0]
        size = info[2]
        Color_Code = val[0]

        # Check if an object with the same Color_Code already exists
       # Unique identifier = EAN + Color Code
        list_of_ean_objects.append({
            "EAN": EAN,
            "Color Code": Color_Code,
            "Size": size
        })

    # temporary_dict[article_no] = list_of_ean_objects

    print(list_of_ean_objects)
    common_po_values = extract_common_po_fields(po_data)
    # print(common_po_values)
    COLOR_SIZE_PDF_PATH = str(get_pdf_directory(
        foldername='data', subfolder='test2', filename='color_size_dialog_tusaha_20241230_053111 (1).pdf'))

    color_size_data = extract_table_data(
        COLOR_SIZE_PDF_PATH, page='all', row_tol=1)
    color_common_value, color_size_info = extract_color_size(color_size_data)
    color_ean_size_info = flatten_color_size_data(
        list_of_ean_objects, color_size_info)
    json = []
    for info in color_ean_size_info:
        combined_object = combine_dicts(
            common_po_values, color_common_value, info)
        json.append(combined_object)

    save_excel_for_pdf('PO10034465-V1_GHK-M000041254', json)
