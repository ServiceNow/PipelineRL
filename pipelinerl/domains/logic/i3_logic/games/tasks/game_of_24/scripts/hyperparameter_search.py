import argparse
import multiprocessing
import os

number_of_data = 30
basic_operators = ["'+'", "'-'", "'*'", "'/'"]
additional_operators = ["%"]
candidates_range = range(1, 10)
num_of_numbers_range = range(3, 6)
result_range = range(20, 30)
operators_num = range(0, 1)


def run_game_process(params):
    num_of_numbers, result, args = params
    candidates = " ".join([str(i) for i in args.candidates_range])
    operators = " ".join(args.basic_operators)
    cmd = f"python -m games.tasks.game_of_24.scripts.game_of_24 --num_of_data {args.number_of_data} --operators {operators} --candidates {candidates} --num_of_numbers {num_of_numbers} --result {result} --max_attempts 1000"
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_data", type=int, default=30)
    parser.add_argument("--basic_operators", type=str, nargs="+", default=basic_operators)
    parser.add_argument("--additional_operators", type=str, nargs="+", default=additional_operators)
    parser.add_argument("--candidates_range", type=int, nargs="+", default=candidates_range)
    parser.add_argument("--num_of_numbers_range", type=int, nargs="+", default=num_of_numbers_range)
    parser.add_argument("--result_range", type=int, nargs="+", default=result_range)
    args = parser.parse_args()
    param_combinations = [(num, res, args) for num in args.num_of_numbers_range for res in args.result_range]
    with multiprocessing.Pool(processes=20) as pool:
        pool.map(run_game_process, param_combinations)
