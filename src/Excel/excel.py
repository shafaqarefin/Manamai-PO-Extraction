
import pandas as pd
from pathlib import Path

# from src.Extraction.tables import get_pdf_json
from utils.dict import combine_dicts, remove_keys_from_dic
from utils.pdf import get_pdf_directory


def save_excel_for_pdf(pdf_id: str, data_list: list, output_dir: str = "output"):
    """
    Save extracted PO data for a single PDF ID as an Excel file.

    Parameters:
    - pdf_id: str, the identifier for the PDF (used as filename)
    - data_list: list of dictionaries containing extracted data
    - output_dir: folder to save Excel file (created if doesn't exist)
    """

    try:
        # No data to save — skip this PDF
        if not data_list:
            print(f"No data to save for PDF ID {pdf_id}.")
            return

        # Create output folder if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert data to DataFrame and save as Excel
        df = pd.DataFrame(data_list)
        file_name = f"{pdf_id}.xlsx"
        save_path = output_path / file_name

        df.to_excel(save_path, index=False)
        print(f"✅ Saved Excel for PDF ID {pdf_id} at {save_path}")

    except Exception as e:
        print(f"❌ Error saving Excel for PDF ID {pdf_id}: {e}")
        raise


def flatten_result_to_individual_objects(values: dict[str, dict[str, list[str]]], header: dict[str, str]) -> dict[str, dict[str, list[dict[str, str]]]]:
    """
    YOUR ORIGINAL FUNCTION - UNCHANGED
    """
    flattened = {}

    for pack_name, fields in values.items():
        # Extract totals (last entries)
        total_pack_ratio = fields.get("Pack Ratio", [None])[-1]
        total_qty = fields.get("Units", [None])[-1]

        # Trim only Pack Ratio and Units
        pack_ratios = fields.get("Pack Ratio", [])[:-1]
        units = fields.get("Units", [])[:-1]

        # Variable fields to align per row
        variable_fields = ["Buyers Colour", "Print", "Size", "SKU"]
        variable_lists = {f: fields.get(f, []) for f in variable_fields}
        variable_lists.update({
            "Pack Ratio": pack_ratios,
            "Units": units
        })

        # Validate lengths
        lengths = [len(v) for v in variable_lists.values()]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"Inconsistent field lengths in {pack_name}: {dict(zip(variable_lists.keys(), lengths))}")

        num_rows = lengths[0] if lengths else 0

        # Shared fields (apply to all individual packs)
        shared_fields = {
            k: (v[0] if isinstance(v, list) and v else "")
            for k, v in fields.items()
            if k not in variable_lists
        }

        individual_packs = []
        for i in range(num_rows):
            row_data = dict(shared_fields)
            for field, values_list in variable_lists.items():
                row_data[field] = values_list[i] if i < len(
                    values_list) else ""
            individual_packs.append(combine_dicts(header, row_data))

        flattened[pack_name] = {
            "Total Pack Ratio": total_pack_ratio or "",
            "Total Qty": total_qty or "",
            "Individual Packs": individual_packs
        }

    return flattened  # <- This returns your original structure


def convert_to_excel_format(flattened_data: dict, end_dict: dict) -> list[dict]:
    """
    NEW FUNCTION: Converts your original nested structure to flat Excel rows
    """
    all_rows = []

    for pack_name, pack_data in flattened_data.items():
        total_pack_ratio = pack_data.get("Total Pack Ratio", "")
        total_qty = pack_data.get("Total Qty", "")
        individual_packs = pack_data.get("Individual Packs", [])

        for i, individual_pack in enumerate(individual_packs):
            row_data = individual_pack.copy()

            # Add pack-level totals ONLY to the first row of each pack
            if i == 0:
                row_data["Total Pack Ratio"] = total_pack_ratio
                row_data["Total Qty"] = total_qty
            else:
                row_data["Total Pack Ratio"] = ""
                row_data["Total Qty"] = ""

            # Add the end_dict values
            row_data.update(end_dict)
            all_rows.append(row_data)

    return all_rows


# In your main code:
def convert_pdf_data_to_excel(hv: dict, nhv: dict, pdf_filename: str, output_dir: str = "output") -> list[dict]:
    """
    Convert extracted PDF data (hv and nhv) to Excel format and save as file.

    Parameters:
    - hv: dict containing header values from PDF extraction
    - nhv: dict containing nested pack values from PDF extraction  
    - pdf_filename: str, the original PDF filename (used for output naming)
    - output_dir: str, directory to save Excel file

    Returns:
    - list[dict]: The flattened Excel-ready data
    """
    # Extract the values you need
    total_units = hv.get('Total Units')
    total_packs = hv.get('Total Packs')
    total_cost_price = hv.get('Total Cost Price')

    # All Header values applicable for all
    end_dict = {
        "Total Units": total_units,
        "Total Packs": total_packs,
        'Total Cost Price': total_cost_price
    }

    # Remove them from hv dictionary
    keys_to_remove = ['Total Units', 'Total Packs', 'Total Cost Price']
    updated_hv = remove_keys_from_dic(hv, keys_to_remove)

    # Process the nested pack data
    flattened = flatten_result_to_individual_objects(nhv, updated_hv)

    # Convert to Excel format
    excel_data = convert_to_excel_format(flattened, end_dict)

    # Save to Excel
    pdf_id = Path(pdf_filename).stem
    save_excel_for_pdf(pdf_id, excel_data, output_dir)

    return excel_data


if __name__ == '__main__':
    PDF_PATH = str(get_pdf_directory('data', 'PO SHEET- BEST & LESS.pdf'))
    # hv, nhv = get_pdf_json(PDF_PATH)

    # Clear and descriptive function name
    # excel_data = convert_pdf_data_to_excel(
    #     hv, nhv, 'PO SHEET- BEST & LESS.pdf')

    # print(f"Processed {len(excel_data)} rows")
