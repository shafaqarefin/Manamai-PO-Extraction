import pandas as pd
import re
import numpy as np


def find_field_location(df: pd.DataFrame, field: str) -> tuple[int, int] | None:
    """
    Find (row, col) of a field in the DataFrame.
    Matches if a line starts with:
        field
        field:
        field:something
        field: something
    (case-insensitive, ignores whitespace/newlines)
    
    Returns:
        (row, col) if found, else None
    """
    field_clean = field.strip().lower()

    try:
        # Normalize all cell text
        df_str = df.astype(str).map(lambda x: x.strip().lower())

        # Split multiline cells
        df_split = df_str.map(
            lambda x: [p.strip() for p in x.split('\n') if p.strip()]
        )

        # Regex pattern: starts with field, optional colon + text
        pattern = re.compile(rf"^{re.escape(field_clean)}\s*:?.*", re.IGNORECASE)

        # Build mask for matches
        mask = df_split.map(lambda parts: any(pattern.match(p) for p in parts))

        rows, cols = np.where(mask.to_numpy())

        # Return first match or None if not found
        if len(rows) == 0:
            return None

        return int(rows[0]), int(cols[0])

    except Exception as e:
        print(f"⚠️ Error while finding '{field}': {e}")
        return None



def find_location_of_all_fields(df: pd.DataFrame, fields: list[str]) -> dict[str, tuple[int, int]]:
    """
    Find (row, col) coordinates of all given fields in the DataFrame.
    Ignores fields not found in df.
    
    Returns:
        dict[field_name, (row, col)]
    """
    results = {}

    for field in fields:
        loc = find_field_location(df, field)
        if loc is not None:
            results[field] = loc

    return results