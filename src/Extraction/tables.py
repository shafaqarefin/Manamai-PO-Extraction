from camelot.io import read_pdf

from src.Preprocessing.preprocess import *

FIELDS_TO_EXTRACT = {
    "Country",
    "Time of Delivery",
    "Invoice Average Price",
    "Planning Markets",
}


def should_stop(cell: str) -> bool:
    """Check if we should stop reading further cells in a column."""
    if not cell or not cell.strip():
        return True
    txt = cell.lower()
    return (
        txt.startswith("by accepting")
        or txt.startswith("(i)")
        or txt.startswith("(ii)")
        or txt.startswith("(iii)")
    )


def findInvoicePricebyCountry(country: str, extracted: dict):
    try:
        for idx, values in enumerate(extracted.get('Country', [])):
            if country in values:
                return extracted['Invoice Average Price'][idx]
        return None
    except Exception as e:
        print(f"❌ Error finding invoice price for {country}: {e}")
        raise


def extract_table_rows(pdf_path: str, page: str = "1"):
    try:
        tables = read_pdf(pdf_path, flavor="stream", pages=page)
        if not tables:
            print(f"❌ No tables found in PDF: {pdf_path}")
            return {}

        df = tables[1].df if len(tables) > 1 else tables[0].df
        df = process_dataframe(df)

        extracted = {field: [] for field in FIELDS_TO_EXTRACT}
        for r_idx, row in df.iterrows():
            for c_idx, val in row.items():
                if val not in FIELDS_TO_EXTRACT:
                    continue

                for down_idx in range(r_idx + 1, len(df)):
                    below = df.iat[down_idx, c_idx]
                    if not isinstance(below, str) or should_stop(below):
                        break

                    if val == "Country":
                        extracted["Country"].append(process_country(below))
                    elif val == "Time of Delivery":
                        extracted["Time of Delivery"].append(
                            process_time_of_delivery(below))
                    elif val == "Planning Markets":
                        extracted["Planning Markets"].append(
                            process_planning_markets(below))
                    elif val == "Invoice Average Price":
                        extracted["Invoice Average Price"].append(
                            process_invoice_price(below))

        return {k: v for k, v in extracted.items() if v}

    except Exception as e:
        print(f"❌ Error extracting table rows from {pdf_path}: {e}")
        raise


def extractTableValues(pdf_path: str):
    try:
        excelObjects = []
        extracted = extract_table_rows(pdf_path)

        if not extracted or 'Planning Markets' not in extracted:
            print(f"No valid data found in {pdf_path}.")
            return []

        for idx, countries in enumerate(extracted.get('Planning Markets', [])):
            for country in countries:
                excelObject = {
                    'Country': country,
                    'Time of Delivery': extracted.get('Time of Delivery', [''])[idx],
                    'Invoice Average Price': findInvoicePricebyCountry(country, extracted),
                }
                excelObjects.append(excelObject)

        return excelObjects

    except Exception as e:
        print(f"❌ Error extracting table values from {pdf_path}: {e}")
        raise
