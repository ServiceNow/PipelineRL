import random
import re
import uuid
from typing import Any, Dict, List, Optional

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.buggy_tables.scripts.buggy_tables_prompt import format_question_template, get_bug_description
from .....games.tasks.buggy_tables.scripts.game_of_buggy_tables_verifier import BuggyTableVerifier
from .....games.tasks.buggy_tables.scripts.handlers.calculate_req_generator import generate_and_execute_query
from .....games.tasks.buggy_tables.scripts.handlers.error_generator import apply_error_makers
from .....games.tasks.buggy_tables.scripts.handlers.tables_handler import (
    generate_random_table,
    transform_to_column_major,
    transform_to_row_major,
)


class GameOfBuggyTables(Game):
    VALID_BUG_TYPES = ["error", "merge_rows", "rotate_columns", "rotate_rows", "add_end_row", "add_end_column", "null"]

    VALID_MAJOR_TYPES = ["col", "row"]

    def __init__(self, name="buggy_tables"):
        super().__init__(name, BuggyTableVerifier)

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        bug_types: List[str] = None,
        num_rows_range: List[int] = None,
        bug_rate_range: List[float] = None,
        bug_weights: Dict[str, float] = None,
    ) -> List[Data]:
        game_data_list = []
        zero_answer_list = []

        question_set = set()

        valid_bug_types = bug_types if bug_types is not None else self.VALID_BUG_TYPES

        min_rows, max_rows = (25, 40) if num_rows_range is None else (num_rows_range[0], num_rows_range[1])

        min_bug_rate, max_bug_rate = (0.05, 0.2) if bug_rate_range is None else (bug_rate_range[0], bug_rate_range[1])

        for bug_type in valid_bug_types:
            if bug_type not in self.VALID_BUG_TYPES:
                raise ValueError(f"Invalid bug type: {bug_type}. Valid types are: {', '.join(self.VALID_BUG_TYPES)}")

        col_major_types = ["error", "merge_rows", "rotate_columns", "rotate_rows"]

        if bug_weights is not None and valid_bug_types:
            filtered_weights = {bt: bug_weights.get(bt, 0) for bt in valid_bug_types}

            filtered_weights = {bt: w for bt, w in filtered_weights.items() if w > 0}

            if filtered_weights:
                weighted_bug_types = list(filtered_weights.keys())
                weights = list(filtered_weights.values())
            else:
                weighted_bug_types = None
                weights = None
        else:
            weighted_bug_types = None
            weights = None

        attempts = 0
        while len(game_data_list) < num_of_questions and attempts < max_attempts:
            attempts += 1

            num_rows = random.randint(min_rows, max_rows)

            if weighted_bug_types and weights:
                bug_type = random.choices(weighted_bug_types, weights=weights, k=1)[0]
            else:
                bug_type = random.choice(valid_bug_types)

            if bug_type in col_major_types:
                major_type = random.choice(self.VALID_MAJOR_TYPES)
            else:
                major_type = "row"

            if bug_type == "null" and major_type == "col":
                major_type = "row"

            bug_rate = round(random.uniform(min_bug_rate, max_bug_rate), 2) if bug_type in ["error", "null"] else None

            result = self._generate_sample(
                num_rows=num_rows, major_type=major_type, bug_type=bug_type, bug_rate=bug_rate
            )

            question_key = result["buggy_table"]
            if question_key not in question_set:
                question_set.add(question_key)

                question = self._create_question(
                    num_rows=num_rows,
                    major_type=major_type,
                    bug_type=bug_type,
                    buggy_table=result["buggy_table"],
                    bug_description=result["bug_description"],
                    queries=result["queries"],
                    target_answer=result["target_answer"],
                )

                metadata = {
                    "trace_id": str(uuid.uuid4()),
                    "num_rows": num_rows,
                    "major_type": major_type,
                    "bug_type": bug_type,
                    "bug_rate": bug_rate,
                    "queries": result["queries"],
                    "buggy_table": result["buggy_table"],
                    "bug_description": result["bug_description"],
                    "target_answer": result["target_answer"],
                    "original_table": result["original_table"],
                    "query_result": result["query_result"],
                }

                game_data = Data(question=question, answer=result["target_answer"], metadata=metadata)

                if result["target_answer"] == "0.00":
                    current_zero_count = sum(1 for data in game_data_list if data.answer == "0.00")

                    if (current_zero_count + 1) / (len(game_data_list) + 1) >= 0.1:
                        zero_answer_list.append(game_data)
                        continue

                game_data_list.append(game_data)

                attempts = 0

        if len(game_data_list) < num_of_questions and zero_answer_list:
            remaining = num_of_questions - len(game_data_list)
            game_data_list.extend(zero_answer_list[:remaining])

        return game_data_list

    def extract_answer(self, test_solution: str) -> str:
        normalized = str(test_solution).strip().lower()

        uncertainty_words = [
            "around",
            "about",
            "between",
            "approximately",
            "roughly",
            "maybe",
            "might",
            "could",
            "possibly",
        ]
        if any(word in normalized for word in uncertainty_words):
            return ""

        numbers = re.findall(r"-?\d+\.\d+|-?\d+", normalized)
        if numbers:
            number = numbers[-1].strip()

            try:
                num_value = float(number)
                return f"{num_value:.2f}"
            except ValueError:
                return number

        indicators = [
            r"answer\s*[=:]\s*([^.,;]+)",
            r"result\s*[=:]\s*([^.,;]+)",
            r"final\s+answer\s*[=:]\s*([^.,;]+)",
            r"the\s+answer\s+is\s*[=:]\s*([^.,;]+)",
            r"calculated\s+value\s*[=:]\s*([^.,;]+)",
            r"value\s*[=:]\s*([^.,;]+)",
            r"output\s*[=:]\s*([^.,;]+)",
            r"is\s*[=:]\s*([^.,;]+)",
            r"equals\s*[=:]\s*([^.,;]+)",
            r"=\s*([^.,;]+)",
        ]

        for pattern in indicators:
            match = re.search(pattern, normalized)
            if match:
                result = match.group(1).strip()

                result = result.rstrip(".,:;")
                return result

        parts = re.split(r"[.,;]", normalized)
        parts = [p.strip() for p in parts if p.strip()]
        if parts:
            return parts[-1].rstrip(".,:;")

        return normalized.rstrip(".,:;")

    def _generate_sample(
        self, num_rows: int, major_type: str, bug_type: str, bug_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        original_df = generate_random_table(num_rows, order_type="random")

        df_copy = original_df.copy()

        if bug_type in ["error", "null"] and bug_rate is not None:
            buggy_df, affected_values, error_positions = apply_error_makers(df_copy, bug_type, error_rate=bug_rate)
        else:
            buggy_df, affected_values, error_positions = apply_error_makers(df_copy, bug_type)

        if major_type == "col":
            buggy_table = transform_to_column_major(buggy_df)
        else:
            buggy_table = transform_to_row_major(buggy_df)

        bug_description = self._generate_bug_description(bug_type, affected_values, error_positions, "column")

        if bug_type == "null":
            query_result = generate_and_execute_query(buggy_df)
        else:
            query_result = generate_and_execute_query(original_df)

        if "conditions" in query_result and isinstance(query_result["conditions"], set):
            query_result["conditions"] = list(query_result["conditions"])

        query = query_result["query"]
        target_answer = query_result["result"]

        if target_answer.replace(".", "", 1).replace("-", "", 1).isdigit():
            try:
                target_answer = f"{float(target_answer):.2f}"
            except ValueError:
                pass

        return {
            "buggy_table": buggy_table,
            "bug_description": bug_description,
            "queries": [query],
            "target_answer": target_answer,
            "original_table": original_df.to_dict(orient="records"),
            "query_result": query_result,
        }

    def _generate_bug_description(
        self, bug_type: str, affected_values: List, error_positions=None, order="column"
    ) -> str:
        language = random.choice(["en", "zh"])

        return get_bug_description(bug_type, affected_values, error_positions, language, order)

    def _create_question(
        self,
        num_rows: int,
        major_type: str,
        bug_type: str,
        buggy_table: str,
        bug_description: str,
        queries: List[str],
        target_answer: str,
    ) -> str:
        if major_type == "row":
            table_format = "row-major"
            table_section = f"{buggy_table}"
        else:
            table_format = "markdown"
            table_section = f"{buggy_table}"

        num_columns = 18

        if any("\u4e00" <= c <= "\u9fff" for c in bug_description):
            language = "zh"
        else:
            language = "en"

        if not queries:
            raise ValueError("queries list cannot be empty")
        question_data = {
            "num_rows": num_rows,
            "num_columns": num_columns,
            "bug_type": bug_type,
            "table_format": table_format,
            "table_section": table_section,
            "bug_description": bug_description,
            "query": queries[0] if len(queries) > 0 else "",
        }

        question = format_question_template(question_data, language)
        answer_format = ""
        return question + answer_format


def main():
    import argparse
    import json
    import pathlib

    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_data", type=int, default=100)
    parser.add_argument("--max_attempts", type=int, default=1000)
    parser.add_argument(
        "--bug_types",
        type=str,
        nargs="+",
        choices=GameOfBuggyTables.VALID_BUG_TYPES,
        help="Specific bug types to generate (defaults to all types)",
    )
    parser.add_argument(
        "--num_rows_range",
        type=int,
        nargs=2,
        default=[25, 40],
        help="Range for number of rows [min, max] (default: [25, 40])",
    )
    parser.add_argument(
        "--bug_rate_range",
        type=float,
        nargs=2,
        default=[0.05, 0.2],
        help="Range for bug rate [min, max] for 'error' and 'null' types (default: [0.05, 0.2])",
    )
    parser.add_argument(
        "--bug_types_ratio",
        type=str,
        default=None,
        help="Ratio for bug types generation in format 'type1:weight1,type2:weight2,...' (e.g., 'error:3,null:2,rotate_rows:1')",
    )
    parser.add_argument(
        "--output_name", type=str, default=None, help="Custom base name for output file (default: 'buggy_tables')"
    )
    args = parser.parse_args()

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    base_filename = args.output_name if args.output_name else "buggy_tables"

    bug_type_suffix = ""
    if args.bug_types:
        bug_type_suffix = f"_{'_'.join(args.bug_types)}"

    num_rows_suffix = ""
    if args.num_rows_range != [25, 40]:
        num_rows_suffix = f"_rows_{args.num_rows_range[0]}_{args.num_rows_range[1]}"

    bug_rate_suffix = ""
    if args.bug_rate_range != [0.05, 0.2]:
        bug_rate_suffix = f"_rate_{args.bug_rate_range[0]}_{args.bug_rate_range[1]}"

    ratio_suffix = ""
    if args.bug_types_ratio:
        ratio_suffix = "_custom_ratio"

    output_file = (
        data_dir
        / f"{base_filename}_{args.num_of_data}{bug_type_suffix}{num_rows_suffix}{bug_rate_suffix}{ratio_suffix}.jsonl"
    )

    bug_weights = None
    if args.bug_types_ratio:
        bug_weights = {}
        try:
            for item in args.bug_types_ratio.split(","):
                bug_type, weight = item.split(":")
                if bug_type in GameOfBuggyTables.VALID_BUG_TYPES:
                    bug_weights[bug_type] = float(weight)
                else:
                    pass

            if not bug_weights:
                bug_weights = None
        except (ValueError, AttributeError):
            bug_weights = None

    game = GameOfBuggyTables()
    game_data_list = game.generate(
        args.num_of_data,
        args.max_attempts,
        args.bug_types,
        num_rows_range=args.num_rows_range,
        bug_rate_range=args.bug_rate_range,
        bug_weights=bug_weights,
    )

    if len(game_data_list) == 0:
        exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for game_data in tqdm(game_data_list, desc="Writing data"):
            f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
