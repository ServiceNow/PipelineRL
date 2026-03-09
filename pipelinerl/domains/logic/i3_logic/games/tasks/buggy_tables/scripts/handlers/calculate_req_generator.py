import random
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from buggy_tables_prompt import format_query_template


def generate_query():
    numeric_columns = [
        "num_steps",
        "study_minutes",
        "exercise_minutes",
        "sleep_minutes",
        "num_messages",
        "num_emails",
        "num_calls",
        "calories_burned",
        "calories_consumed",
        "num_meetings",
        "coding_minutes",
        "num_tasks_completed",
        "water_intake_ml",
        "phone_screen_time_minutes",
        "music_listening_minutes",
    ]
    cols_to_compare = random.sample(numeric_columns, 2)
    basic_stats = ["mean", "median", "stdev", "sum", "max", "min", "variance"]
    all_stats = basic_stats
    stat_type = random.choice(all_stats)
    filter_conditions = {}
    for column in numeric_columns:
        basic_operators = [">", "<", ">=", "<=", "==", "!="]
        range_operator = ["between"]
        operators = basic_operators + (range_operator if random.random() < 0.2 else [])
        operator = random.choice(operators)
        if column == "num_steps":
            base_value = random.randint(10000, 20000)
            value = [base_value - 1000, base_value + 1000] if operator == "between" else base_value
        elif column == "study_minutes":
            base_value = random.randint(15, 30)
            value = [base_value - 5, base_value + 5] if operator == "between" else base_value
        elif column == "exercise_minutes":
            base_value = random.randint(40, 60)
            value = [base_value - 10, base_value + 10] if operator == "between" else base_value
        elif column == "sleep_minutes":
            base_value = random.randint(300, 400)
            value = [base_value - 30, base_value + 30] if operator == "between" else base_value
        elif column == "num_messages":
            base_value = random.randint(3, 8)
            value = [base_value - 1, base_value + 1] if operator == "between" else base_value
        elif column == "num_emails":
            base_value = random.randint(1, 5)
            value = [base_value - 1, base_value + 1] if operator == "between" else base_value
        elif column == "num_calls":
            base_value = random.randint(1, 4)
            value = [base_value - 1, base_value + 1] if operator == "between" else base_value
        elif column == "calories_burned":
            base_value = random.randint(600, 1000)
            value = [base_value - 100, base_value + 100] if operator == "between" else base_value
        elif column == "calories_consumed":
            base_value = random.randint(1600, 2000)
            value = [base_value - 200, base_value + 200] if operator == "between" else base_value
        elif column == "num_meetings":
            base_value = random.randint(1, 4)
            value = [base_value - 1, base_value + 1] if operator == "between" else base_value
        elif column == "coding_minutes":
            base_value = random.randint(15, 30)
            value = [base_value - 5, base_value + 5] if operator == "between" else base_value
        elif column == "num_tasks_completed":
            base_value = random.randint(2, 5)
            value = [base_value - 1, base_value + 1] if operator == "between" else base_value
        elif column == "water_intake_ml":
            base_value = random.randint(2000, 3000)
            value = [base_value - 200, base_value + 200] if operator == "between" else base_value
        elif column == "phone_screen_time_minutes":
            base_value = random.randint(20, 30)
            value = [base_value - 5, base_value + 5] if operator == "between" else base_value
        elif column == "music_listening_minutes":
            base_value = random.randint(20, 30)
            value = [base_value - 5, base_value + 5] if operator == "between" else base_value
        filter_conditions[column] = {"op": operator, "value": value}
    all_columns = list(filter_conditions.keys())
    num_conditions = random.choice([2, 3])
    filter_cols = random.sample(all_columns, num_conditions)
    query_data = {
        "cols_to_compare": cols_to_compare,
        "stat_type": stat_type,
        "conditions": {col: filter_conditions[col] for col in filter_cols},
    }
    language = random.choice(["en", "zh"])
    query = format_query_template(query_data, language)
    return {
        "query": query,
        "cols_to_compare": cols_to_compare,
        "stat_type": stat_type,
        "conditions": {col: filter_conditions[col] for col in filter_cols},
    }


def execute_query(df, query_info):
    cols_to_compare = query_info["cols_to_compare"]
    stat_type = query_info["stat_type"]
    conditions = query_info["conditions"]
    available_columns = [col for col in cols_to_compare if col in df.columns]
    if len(available_columns) < 2:
        return {
            "query": query_info["query"],
            "result": "0",
            "filtered_rows": 0,
            "total_rows": len(df),
            "conditions": {f"{col} {conditions[col]['op']} {conditions[col]['value']}" for col in conditions.keys()},
            "columns_compared": cols_to_compare,
            "statistic": stat_type,
        }
    filtered_df = df.copy()
    for condition, cond_info in conditions.items():
        if condition not in df.columns:
            continue
        operator = cond_info["op"]
        value = cond_info["value"]
        try:
            if operator == "between":
                if isinstance(value, list) and len(value) == 2:
                    filtered_df = filtered_df[
                        filtered_df[condition].notna()
                        & (filtered_df[condition] >= value[0])
                        & (filtered_df[condition] <= value[1])
                    ]
            else:
                op_map = {">": "__gt__", "<": "__lt__", ">=": "__ge__", "<=": "__le__", "==": "__eq__", "!=": "__ne__"}
                if operator in op_map:
                    if operator == "!=":
                        filtered_df = filtered_df[
                            filtered_df[condition].notna() & getattr(filtered_df[condition], op_map[operator])(value)
                        ]
                    else:
                        filtered_df = filtered_df[
                            filtered_df[condition].notna() & getattr(filtered_df[condition], op_map[operator])(value)
                        ]
        except Exception:
            filtered_df = pd.DataFrame(columns=filtered_df.columns)
    result = "0"
    try:
        if len(filtered_df) > 0:
            col1, col2 = cols_to_compare

            def calculate_basic_stat(series, stat_type):
                if len(series) == 0 or series.isna().all():
                    return 0
                if stat_type == "mean":
                    return series.mean()
                elif stat_type == "median":
                    return series.median()
                elif stat_type == "stdev":
                    return 0 if len(series.dropna()) <= 1 else series.std()
                elif stat_type == "sum":
                    return series.sum()
                elif stat_type == "max":
                    return series.max()
                elif stat_type == "min":
                    return series.min()
                elif stat_type == "variance":
                    return 0 if len(series.dropna()) <= 1 else series.var()
                return 0

            if len(filtered_df[col1].dropna()) == 0 or len(filtered_df[col2].dropna()) == 0:
                result = "0.00"
            elif stat_type in ["stdev", "variance"] and (
                len(filtered_df[col1].dropna()) <= 1 or len(filtered_df[col2].dropna()) <= 1
            ):
                result = "0.00"
            else:
                stat1 = calculate_basic_stat(filtered_df[col1], stat_type)
                stat2 = calculate_basic_stat(filtered_df[col2], stat_type)
                if pd.notna(stat1) and pd.notna(stat2):
                    result = f"{abs(stat1 - stat2):.2f}"
    except Exception:
        pass
    if result == "0":
        result = "0.00"
    elif result.replace(".", "", 1).replace("-", "", 1).isdigit():
        try:
            result = f"{float(result):.2f}"
        except ValueError:
            pass
    return {
        "query": query_info["query"],
        "result": result,
        "filtered_rows": len(filtered_df),
        "total_rows": len(df),
        "conditions": {f"{col} {conditions[col]['op']} {conditions[col]['value']}" for col in conditions.keys()},
        "columns_compared": cols_to_compare,
        "statistic": stat_type,
    }


def generate_and_execute_query(df):
    query_info = generate_query()
    result_info = execute_query(df, query_info)
    return result_info
