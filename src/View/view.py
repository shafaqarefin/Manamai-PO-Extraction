"""View module for displaying DataFrames."""
import pandas as pd


def display_single_dataframe(df: pd.DataFrame, label: str = "DataFrame",
                             show_shape: bool = True) -> None:
    """
    Display a single DataFrame.

    Args:
        df: DataFrame to display
        label: Label for the DataFrame
        show_shape: Whether to show dimensions
    """
    print(f"\n{'=' * 80}")
    print(f"📊 {label}")
    if show_shape:
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print('=' * 80)
    print(df)
    print('=' * 80 + '\n')


def display_flat_dict(dataframes: dict, title: str = "DataFrames",
                      show_shape: bool = True) -> None:
    """
    Display DataFrames from a flat dictionary.

    Args:
        dataframes: {"label1": df1, "label2": df2}
        title: Title for the entire display
        show_shape: Whether to show dimensions
    """
    print(f"\n{'#' * 80}")
    print(f"📚 {title}")
    print('#' * 80 + '\n')

    for label, df in dataframes.items():
        if isinstance(df, pd.DataFrame):
            display_single_dataframe(df, label=label, show_shape=show_shape)


def display_nested_dict(data: dict, title: str = "DataFrames",
                        show_shape: bool = False,) -> None:
    """
    Display DataFrames from nested dictionary.

    Args:
        data: {"Pack 1": {"group": df1, "unique": df2}}
        title: Title for the entire display
        show_shape: Whether to show dimensions
    """
    print(f"\n{'#' * 80}")
    print(f"📚 {title}")
    print('#' * 80 + '\n')

    for main_key, sub_data in data.items():
        print(f"\n{'─' * 80}")
        print(f"📦 {main_key}")
        print('─' * 80)

        for sub_key, df in sub_data.items():
            for label, table in df.items():
                print(table)
