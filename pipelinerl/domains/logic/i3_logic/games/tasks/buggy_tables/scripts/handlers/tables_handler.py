import random
from datetime import datetime, timedelta

import pandas as pd


def generate_random_table(row_num, order_type="random"):
    all_columns = {
        "date": lambda n: [
            (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d") for _ in range(n)
        ],
        "num_steps": lambda n: [random.randint(1000, 20000) for _ in range(n)],
        "study_minutes": lambda n: [random.randint(0, 360) for _ in range(n)],
        "exercise_minutes": lambda n: [random.randint(0, 120) for _ in range(n)],
        "sleep_minutes": lambda n: [random.randint(300, 600) for _ in range(n)],
        "bed_time": lambda n: [f"{random.randint(21, 23)}:{random.randint(0, 59):02d}" for _ in range(n)],
        "num_messages": lambda n: [random.randint(0, 200) for _ in range(n)],
        "num_emails": lambda n: [random.randint(0, 50) for _ in range(n)],
        "num_calls": lambda n: [random.randint(0, 15) for _ in range(n)],
        "calories_burned": lambda n: [random.randint(1500, 3500) for _ in range(n)],
        "weekday": lambda n: [random.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]) for _ in range(n)],
        "calories_consumed": lambda n: [random.randint(1200, 3000) for _ in range(n)],
        "num_meetings": lambda n: [random.randint(0, 8) for _ in range(n)],
        "coding_minutes": lambda n: [random.randint(0, 480) for _ in range(n)],
        "num_tasks_completed": lambda n: [random.randint(0, 20) for _ in range(n)],
        "water_intake_ml": lambda n: [random.randint(500, 3000) for _ in range(n)],
        "phone_screen_time_minutes": lambda n: [random.randint(30, 360) for _ in range(n)],
        "music_listening_minutes": lambda n: [random.randint(0, 240) for _ in range(n)],
    }
    selected_columns = list(all_columns.keys())
    data = {col: all_columns[col](row_num) for col in selected_columns}
    df = pd.DataFrame(data)
    if order_type == "ascending" and "date" in df.columns:
        df = df.sort_values("date")
    elif order_type == "descending" and "date" in df.columns:
        df = df.sort_values("date", ascending=False)
    return df


def transform_to_column_major(df):
    has_extra_cols = any((col.startswith("extra_col_") for col in df.columns))
    if has_extra_cols:
        original_cols = [col for col in df.columns if not col.startswith("extra_col_")]
        df_display = df[original_cols].copy()
    else:
        df_display = df.copy()
    header = "| " + " | ".join(df_display.columns) + " |"
    separator = "| " + " | ".join(["---" for _ in range(len(df_display.columns))]) + " |"
    rows = []
    for _, row in df_display.iterrows():
        values = [str(val) if pd.notna(val) else "" for val in row.values]
        rows.append("| " + " | ".join(values) + " |")
    markdown_table = "\n".join([header, separator] + rows)
    return markdown_table


def transform_to_row_major(df):
    original_cols = [col for col in df.columns if not col.startswith("extra_col_")]
    extra_cols = [col for col in df.columns if col.startswith("extra_col_")]
    row_major_list = list(original_cols)
    for i in range(len(df)):
        row = df.iloc[i]
        for col in original_cols:
            val = row[col]
            if not pd.isna(val):
                row_major_list.append(str(val))
        for col in extra_cols:
            val = row[col]
            if not pd.isna(val):
                row_major_list.append(str(val))
    row_major_str = "[" + ", ".join(row_major_list) + "]"
    return row_major_str


if __name__ == "__main__":
    row_num = 5
    df = generate_random_table(row_num)
    df.to_csv("table.csv", index=False)
    major_type = "row"
    if major_type == "col":
        output = transform_to_column_major(df)
    elif major_type == "row":
        output = transform_to_row_major(df)
    with open(f"{major_type}_table.md", "w") as f:
        f.write(output)
