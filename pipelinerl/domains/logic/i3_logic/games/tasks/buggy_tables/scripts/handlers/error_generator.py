import random
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def make_error(
    df: pd.DataFrame, error_rate: float, error_order: Optional[str] = None
) -> Tuple[pd.DataFrame, List, List[Tuple[int, int]]]:
    if error_order is None:
        error_order = "ERROR"
    df_copy = df.copy(deep=True)
    total_cells = df.size
    num_errors = int(total_cells * error_rate)
    rows, cols = df_copy.shape
    positions = [(i, j) for i in range(rows) for j in range(cols)]
    error_positions = random.sample(positions, num_errors)
    original_values = []
    for col in df_copy.columns:
        df_copy[col] = df_copy[col].astype(object)
    for pos in error_positions:
        i, j = pos
        original_value = df_copy.iloc[i, j]
        original_values.append(original_value)
        df_copy.iloc[i, j] = error_order
    return (df_copy, original_values, error_positions)


def merge_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
    df_copy = df.copy(deep=True)
    rows = df_copy.shape[0]
    merged_rows = []
    for i in range(0, rows, 2):
        if i + 1 < rows:
            merged_row = []
            for j in range(df_copy.shape[1]):
                merged_row.append(f"{df_copy.iloc[i, j]} && {df_copy.iloc[i + 1, j]}")
            merged_rows.append(merged_row)
        else:
            merged_rows.append(df_copy.iloc[i].tolist())
    return (pd.DataFrame(merged_rows, columns=df_copy.columns), [])


def rotate_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
    df_copy = df.copy(deep=True)
    for col in df_copy.columns:
        df_copy[col] = df_copy[col].astype(object)
    for i in range(df_copy.shape[1]):
        col_values = df_copy.iloc[:, i].tolist()
        rotated_values = col_values[-i:] + col_values[:-i] if i > 0 else col_values
        df_copy.iloc[:, i] = rotated_values
    return (df_copy, [])


def rotate_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
    df_copy = df.copy(deep=True)
    for col in df_copy.columns:
        df_copy[col] = df_copy[col].astype(object)
    for i in range(df_copy.shape[0]):
        row_values = df_copy.iloc[i, :].tolist()
        rotated_values = row_values[-i:] + row_values[:-i] if i > 0 else row_values
        df_copy.iloc[i, :] = rotated_values
    return (df_copy, [])


def add_end_row(df: pd.DataFrame, min_val: float = -10, max_val: float = 10) -> Tuple[pd.DataFrame, List]:
    df_copy = df.copy(deep=True)
    rows, cols = df_copy.shape
    new_cols = cols + rows - 1
    original_columns = list(df_copy.columns)
    new_columns = original_columns.copy()
    for i in range(cols, new_cols):
        new_columns.append(f"extra_col_{i - cols + 1}")

    def generate_random_for_column(column_name, column_values):
        if column_name == "date":
            return [(datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")]
        elif column_name == "bed_time":
            return [f"{random.randint(21, 23)}:{random.randint(0, 59):02d}"]
        elif column_name == "weekday":
            return [random.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])]
        elif "minutes" in column_name.lower():
            non_null_values = [v for v in column_values if pd.notna(v)]
            if non_null_values:
                min_minutes = max(0, min(non_null_values) * 0.8)
                max_minutes = max(non_null_values) * 1.2
                return [round(random.uniform(min_minutes, max_minutes), 2)]
            else:
                return [round(random.uniform(0, 240), 2)]
        elif "num_" in column_name.lower():
            non_null_values = [v for v in column_values if pd.notna(v)]
            if non_null_values:
                min_num = max(0, min(non_null_values) * 0.8)
                max_num = max(non_null_values) * 1.2
                return [round(random.uniform(min_num, max_num), 2)]
            else:
                return [round(random.uniform(0, 50), 2)]
        elif "calories" in column_name.lower():
            non_null_values = [v for v in column_values if pd.notna(v)]
            if non_null_values:
                min_cal = max(0, min(non_null_values) * 0.8)
                max_cal = max(non_null_values) * 1.2
                return [round(random.uniform(min_cal, max_cal), 2)]
            else:
                return [round(random.uniform(1000, 3000), 2)]
        else:
            non_null_values = [v for v in column_values if pd.notna(v)]
            if non_null_values and all((isinstance(v, (int, float)) for v in non_null_values)):
                min_val = min(non_null_values) * 0.8
                max_val = max(non_null_values) * 1.2
                return [round(random.uniform(min_val, max_val), 2)]
            else:
                return [round(random.uniform(-10, 10), 2)]

    new_df = pd.DataFrame(columns=new_columns, index=range(rows))
    for i in range(rows):
        for j in range(cols):
            new_df.iloc[i, j] = df_copy.iloc[i, j]
    if not original_columns:
        return (new_df, [])
    for i in range(rows):
        num_extra_values = i
        for j in range(num_extra_values):
            col_idx = cols + j
            # col_name = new_columns[col_idx]
            if not original_columns:
                continue
            ref_col_name = original_columns[0]
            ref_col_values = df_copy[ref_col_name].values.tolist()
            random_val = generate_random_for_column(ref_col_name, ref_col_values)[0]
            new_df.iloc[i, col_idx] = random_val
    return (new_df, [])


def add_end_column(df: pd.DataFrame, min_val: float = -10, max_val: float = 10) -> Tuple[pd.DataFrame, List]:
    df_copy = df.copy(deep=True)
    rows, cols = df_copy.shape
    # new_rows = rows + cols - 1

    def generate_random_for_column(column_name, column_values):
        if column_name == "date":
            return [(datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")]
        elif column_name == "bed_time":
            return [f"{random.randint(21, 23)}:{random.randint(0, 59):02d}"]
        elif column_name == "weekday":
            return [random.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])]
        elif "minutes" in column_name.lower():
            non_null_values = [v for v in column_values if pd.notna(v)]
            if non_null_values:
                min_minutes = max(0, min(non_null_values) * 0.8)
                max_minutes = max(non_null_values) * 1.2
                return [round(random.uniform(min_minutes, max_minutes), 2)]
            else:
                return [round(random.uniform(0, 240), 2)]
        elif "num_" in column_name.lower():
            non_null_values = [v for v in column_values if pd.notna(v)]
            if non_null_values:
                min_num = max(0, min(non_null_values) * 0.8)
                max_num = max(non_null_values) * 1.2
                return [round(random.uniform(min_num, max_num), 2)]
            else:
                return [round(random.uniform(0, 50), 2)]
        elif "calories" in column_name.lower():
            non_null_values = [v for v in column_values if pd.notna(v)]
            if non_null_values:
                min_cal = max(0, min(non_null_values) * 0.8)
                max_cal = max(non_null_values) * 1.2
                return [round(random.uniform(min_cal, max_cal), 2)]
            else:
                return [round(random.uniform(1000, 3000), 2)]
        else:
            non_null_values = [v for v in column_values if pd.notna(v)]
            if non_null_values and all((isinstance(v, (int, float)) for v in non_null_values)):
                min_val = min(non_null_values) * 0.8
                max_val = max(non_null_values) * 1.2
                return [round(random.uniform(min_val, max_val), 2)]
            else:
                return [round(random.uniform(-10, 10), 2)]

    for j in range(cols):
        col_name = df_copy.columns[j]
        col_values = df_copy.iloc[:, j].values.tolist()
        num_extra_values = j
        for i in range(num_extra_values):
            row_idx = rows + i
            if row_idx >= len(df_copy):
                new_row = pd.Series([np.nan] * cols, index=df_copy.columns)
                df_copy = pd.concat([df_copy, pd.DataFrame([new_row])], ignore_index=True)
            random_val = generate_random_for_column(col_name, col_values)[0]
            df_copy.iloc[row_idx, j] = random_val
    return (df_copy, [])


def add_null_values(df: pd.DataFrame, error_rate: float = 0.1) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    df_copy = df.copy(deep=True)
    rows, cols = df_copy.shape
    total_cells = rows * cols
    num_nulls = int(total_cells * error_rate)
    all_coordinates = [(i, j) for i in range(rows) for j in range(cols)]
    coordinates = random.sample(all_coordinates, min(num_nulls, len(all_coordinates)))
    for row, col in coordinates:
        if 0 <= row < df_copy.shape[0] and 0 <= col < df_copy.shape[1]:
            df_copy.iloc[row, col] = None
    formatted_coordinates = [(int(row), int(col)) for row, col in coordinates]
    return (df_copy, formatted_coordinates)


def apply_error_makers(
    df: pd.DataFrame,
    error_type: str,
    error_rate: float = 0.1,
    error_order: str = "ERROR",
    null_coordinates: List[Tuple[int, int]] = None,
    random_value_range: Tuple[float, float] = (0, 100),
) -> Tuple[pd.DataFrame, List, Optional[List[Tuple[int, int]]]]:
    min_val, max_val = random_value_range
    if error_type == "error":
        df_with_errors, original_values, error_positions = make_error(df, error_rate, error_order)
        return (df_with_errors, original_values, error_positions)
    elif error_type == "merge_rows":
        return merge_rows(df) + (None,)
    elif error_type == "rotate_columns":
        return rotate_columns(df) + (None,)
    elif error_type == "rotate_rows":
        return rotate_rows(df) + (None,)
    elif error_type == "add_end_row":
        return add_end_row(df, min_val, max_val) + (None,)
    elif error_type == "add_end_column":
        return add_end_column(df, min_val, max_val) + (None,)
    elif error_type == "null":
        df_with_nulls, null_positions = add_null_values(df, error_rate)
        return (df_with_nulls, null_positions, null_positions)
    else:
        raise ValueError(f"不支持的错误类型: {error_type}")
