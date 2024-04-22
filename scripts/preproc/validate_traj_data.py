
import pandas as pd
import os
from typing import List

def check_csv_specs(file_path: str, expected_rows: int, expected_columns: int) -> bool:
    """
    Check if a CSV file meets the expected number of rows and columns.
    
    Args:
        file_path (str): The path to the CSV file.
        expected_rows (int): The expected number of rows.
        expected_columns (int): The expected number of columns.

    Returns:
        bool: True if specs are met, False otherwise.
    """
    df = pd.read_csv(file_path)
    return df.shape[0] == expected_rows and df.shape[1] == expected_columns

def validate_dataset_specs(dir_path: str, expected_rows: int = 50, expected_columns: int = 46) -> List[str]:
    """
    Validate if all CSV files in a directory meet the dataset specifications.

    Args:
        dir_path (str): The path to the directory containing CSV files.
        expected_rows (int, optional): The expected number of rows in each CSV file. Defaults to 50.
        expected_columns (int, optional): The expected number of columns in each CSV file. Defaults to 46.

    Returns:
        List[str]: List of file paths that don't meet the specifications.
    """
    invalid_files = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(dir_path, filename)
            if not check_csv_specs(file_path, expected_rows, expected_columns):
                invalid_files.append(file_path)
    return invalid_files

if __name__ == "__main__":
    dataset_dirs = [
        "teamtrack_traj/train",
        "teamtrack_traj/val",
        "teamtrack_traj/test",
        "jleague/train",
        "jleague/val",
        "jleague/test",
        "stats_perform/train",
        "stats_perform/val",
        "stats_perform/test",
    ]

    for dir_path in dataset_dirs:
        invalid_files = validate_dataset_specs(dir_path)
        if invalid_files:
            print(f"Files in {dir_path} that don't meet specs: {invalid_files}")
        else:
            print(f"All files in {dir_path} meet the specifications.")
