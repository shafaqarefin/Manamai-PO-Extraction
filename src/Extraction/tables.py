from camelot.io import read_pdf
from numpy import extract
from utils.pdfPath import get_pdf_path


def extract_table_rows(pdf_path: str, page: str = "1"):
    tables = read_pdf(pdf_path, flavor="stream", pages=page)
    df = tables[1].df

    # Strip whitespace from each cell
    for r_idx, row in df.iterrows():
        for c_idx, val in row.items():
            if isinstance(val, str):
                df.iat[r_idx, c_idx] = val.strip()

    fields_to_extract = {
        "Order No",
        "Country",
        "Product Description",
        "Season",
        "Type of Construction",
        "No. of Pieces",
        "Sales Mode",
        "Time of Delivery",
        "Invoice Average Price",
        "Planning Markets"
    }

    extracted = {field: [] for field in fields_to_extract}

    def should_stop(cell: str) -> bool:
        if not cell or not cell.strip():
            return True
        txt = cell.lower()
        return (
            txt.startswith("by accepting")
            or txt.startswith("(i)")
            or txt.startswith("(ii)")
            or txt.startswith("(iii)")
        )

    for r_idx, row in df.iterrows():
        for c_idx, val in row.items():
            if val in fields_to_extract:
                col_values = []
                for down_idx in range(r_idx + 1, len(df)):
                    below = df.iat[down_idx, c_idx]
                    if not isinstance(below, str) or should_stop(below):
                        break

                    # Handle Country normally
                    if val == "Country":
                        group = [p.strip() for p in below.split(",")]
                        extracted["Country"].append(group)

                    # Handle Planning Markets
                    elif val == "Planning Markets":
                        if "," in below:
                            parts = []
                            for item in below.split(","):
                                first_word = item.strip().split()[0]
                                parts.append(first_word)
                            extracted["Planning Markets"].append(parts)
                        else:
                            first_word = below.strip().split()[0]
                            extracted["Planning Markets"].append(
                                [first_word])

                    # Other fields
                    else:
                        extracted[val].append(below)

    return {k: v for k, v in extracted.items() if v}


def createNonTableObject(extracted: dict):
    excelObjects = []

    # Loop over all rows in Planning Markets
    for idx, countries in enumerate(extracted['Planning Markets']):
        for country in countries:
            excelObject = {}
            excelObject['Country'] = country
            excelObject['Time of Delivery'] = extracted['Time of Delivery'][idx]
            excelObject['Invoice Average Price'] = findInvoicePricebyCountry(
                country, extracted)
            excelObjects.append(excelObject)

    return excelObjects


def findInvoicePricebyCountry(country: str, extracted: dict):
    for idx, values in enumerate(extracted['Country']):
        if country in values:
            invoiceValue = extracted['Invoice Average Price'][idx]
            return invoiceValue


if __name__ == "__main__":
    path = get_pdf_path(
        "416605_PurchaseOrder_Supplier_20250915_020609.pdf", "data")
    result = extract_table_rows(path, page="1")
    print(createNonTableObject(result))
    # for k, v in result.items():
    #     print(f"{k}: {v}")
