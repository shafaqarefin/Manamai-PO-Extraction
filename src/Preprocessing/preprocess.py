from utils.formatDate import formatDate


def process_dataframe(df):
    """Strip whitespace from all string cells in a dataframe."""
    for r_idx, row in df.iterrows():
        for c_idx, val in row.items():
            if isinstance(val, str):
                df.iat[r_idx, c_idx] = val.strip()
    return df


def process_country(cell_value: str):
    return [p.strip() for p in cell_value.split(",")]


def process_time_of_delivery(cell_value: str):
    return formatDate(cell_value)


def process_planning_markets(cell_value: str):
    if "," in cell_value:
        return [item.strip().split()[0] for item in cell_value.split(",")]
    else:
        return [cell_value.strip().split()[0]]


def process_invoice_price(cell_value: str):
    return cell_value.strip().split()[0]
