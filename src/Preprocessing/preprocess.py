"""Preprocessing module for DataFrame manipulation."""
import re
import pandas as pd
from utils.field import find_field_location


def split_combined_columns_df(df: pd.DataFrame, char: str = '\n') -> pd.DataFrame:
    """
    Splits cells containing a specified character (e.g., newline '\n') across multiple
    new columns, fixing column misalignment typical in table extraction.

    Iterates through the DataFrame columns. If any cell in a column contains the
    `char` delimiter, the cell's content is split, and the resulting parts are
    injected as new columns immediately to the right, shifting subsequent columns.

    Args:
        df (pd.DataFrame): The input DataFrame, typically from table extraction.
        char (str): The character used to delimit combined values (e.g., '\\n').
                    Defaults to '\\n'.

    Returns:
        pd.DataFrame: A new DataFrame with split columns and aligned data.
                      NaN values resulting from the split are converted to "".
    """
    if not char:
        # Note: The default argument now handles this, but keeping a check is safe.
        print("Error: Delimiter character (char) cannot be None.")
        return df

    new_data = {}
    col_offset = 0

    # Iterate through the columns of the original DataFrame
    for col in df.columns:
        # Get the column data as a Series
        col = int(col)
        current_series = df[col]

        # Check if ANY element in the column contains the delimiter character
        # We use .astype(str) to safely check for the substring, handling NaNs
        if current_series.astype(str).str.contains(char, regex=False).any():

            # 1. Split the column data. `expand=True` creates a temporary DataFrame
            #    where each part of the split is in a new column (0, 1, 2, ...).
            split_data = current_series.str.split(char, expand=True)

            # --- Feature: Replace NaN/None in split parts with "" ---
            # Fill NaN values in the split data with empty strings. This ensures
            # that where a split part doesn't exist (e.g., only 3 values were split
            # but the max was 5), the cell is represented by "" instead of NaN.
            split_data = split_data.fillna("")

            # 2. Add the first part (the 'head') to the new DataFrame structure
            #    This part replaces the original column's position (col + current offset).
            new_col_key = col + col_offset
            new_data[new_col_key] = split_data[0]

            # 3. Add the remaining split parts as new columns (injected beside)
            #    Iterate from the second part (index 1) up to the max split.
            for i in range(1, split_data.shape[1]):
                col_offset += 1  # Increment offset for the new injected column
                injected_col_key = col + col_offset
                new_data[injected_col_key] = split_data[i]

        else:
            # If no split is needed, just add the original column's data.
            # Convert any existing None/NaN in the original series to "" for consistency
            # if we expect string data, otherwise we rely on the final DataFrame construction.
            # Keeping the original values (including NaN) here is usually safer.
            new_col_key = col + col_offset
            new_data[new_col_key] = current_series.fillna("")

    # Create the final DataFrame from the new_data dictionary
    # The keys (col + col_offset) are already unique and in the correct order.
    new_df = pd.DataFrame(new_data)

    # Final touch: Reset column names to a clean integer sequence (0, 1, 2, ...)
    new_df.columns = range(new_df.shape[1])

    return new_df


def split_by_pack_and_column(df: pd.DataFrame, column_field="Buyers Colour") -> dict[str, dict[str, pd.DataFrame]]:
    """
    Splits the DataFrame first by Pack (rows) and then each pack by column containing `column_field` (regex, case-insensitive).
    Returns nested dict: {pack_name: {"group": df_before, "unique": df_after}}
    """
    # Step 1: Find rows where first column contains "Pack X"
    pack_rows = df[df.iloc[:, 0].str.contains(
        r"Page \d+", na=False)].index.tolist()
    result = {}

    for idx, start_row in enumerate(pack_rows):
        pack_name = df.iloc[start_row, 0]  # "Pack 1", etc.
        end_row = pack_rows[idx + 1] if idx + 1 < len(pack_rows) else len(df)
        pack_df = df.iloc[start_row:end_row].copy()

        # Step 2: Find column index where any cell matches column_field regex
        col_idx = None
        pattern = re.compile(column_field, flags=re.IGNORECASE)
        for i, col in enumerate(pack_df.columns):
            if pack_df[col].astype(str).str.contains(pattern).any():
                col_idx = i
                break

        # Step 3: Split into 'group' (before column) and 'vertical_fields' (column_field onwards)
        if col_idx is None:
            split_dict = {"horizontal_fields": drop_empty_columns_rows(pack_df),
                          "vertical_fields": drop_empty_columns_rows(pd.DataFrame())}
        else:
            split_dict = {
                "horizontal_fields": drop_empty_columns_rows(pack_df.iloc[:, :col_idx].copy()),
                "vertical_fields": drop_empty_columns_rows(pack_df.iloc[:, col_idx:].copy())
            }

        result[pack_name] = split_dict

    return result


def get_data_by_pattern(df: pd.DataFrame, pattern: str, pattern2: str | None = None, mode: str = "before") -> pd.DataFrame:
    """
    Get data either before, after, or between patterns in a DataFrame.

    Args:
        df: The DataFrame to search.
        pattern: First pattern to locate (regex or plain text).
        pattern2: Optional second pattern (used only if mode='between').
        mode:
            - "before": returns data before the first pattern.
            - "after": returns data after the first pattern.
            - "between": returns data between pattern and pattern2.

    Returns:
        pd.DataFrame: Sliced DataFrame based on mode.
    """
    try:
        if df is None or df.empty:
            print("⚠️ DataFrame is empty or None.")
            return pd.DataFrame()

        if mode not in ["before", "after", "between"]:
            raise ValueError("mode must be 'before', 'after', or 'between'")

        # Find first pattern location
        loc1 = find_field_location(df, pattern)
        if not loc1:
            print(f"⚠️ Pattern '{pattern}' not found in DataFrame.")
            return pd.DataFrame()
        r, c = loc1

        if mode == "before":
            result_df = df.iloc[:r, :c].copy()
        elif mode == "after":
            result_df = df.iloc[r:, c:].copy()
        else:  # mode == "between"
            if not pattern2:
                raise ValueError(
                    "pattern2 must be provided when mode='between'")

            loc2 = find_field_location(df, pattern2)
            if not loc2:
                print(
                    f"⚠️ Second pattern '{pattern2}' not found in DataFrame.")
                return pd.DataFrame()

            r1, c1 = loc2
            # print(r, c)
            # print(r1, c1)

            # Ensure r2/c2 are after r/c; otherwise return empty
            # if r2 >= r or c2 > c:
            #     print(
            #         f"⚠️ Second pattern '{pattern2}' occurs before first pattern '{pattern}'")
            #     return pd.DataFrame()

            result_df = df.iloc[r:r1, c:].copy()

        return drop_empty_columns_rows(result_df)

    except Exception as e:
        print(f"❌ Error processing pattern '{pattern}': {e}")
        return pd.DataFrame()


def drop_empty_columns_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns and rows that are completely empty (all NaN or empty strings).

    Args:
        df: The DataFrame to clean

    Returns:
        DataFrame with empty rows and columns removed
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()

    # Replace empty strings with NaN for easier processing
    cleaned_df = cleaned_df.replace('', pd.NA)
    # Also handle whitespace-only cells
    cleaned_df = cleaned_df.replace(' ', pd.NA)

    # Drop columns where ALL values are NaN/empty
    cleaned_df = cleaned_df.dropna(axis=1, how='all')

    # Drop rows where ALL values are NaN/empty
    cleaned_df = cleaned_df.dropna(axis=0, how='all')

    # Reset index to have clean sequential indices
    cleaned_df = cleaned_df.reset_index(drop=True)

    return cleaned_df
