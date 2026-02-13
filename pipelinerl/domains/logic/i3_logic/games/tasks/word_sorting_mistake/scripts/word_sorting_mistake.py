import argparse
import json
import pathlib
import random
from collections import defaultdict

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.word_sorting_mistake.scripts.word_sorting_mistake_prompt import (
    chinese_prompts,
    english_prompts,
)
from .....games.tasks.word_sorting_mistake.scripts.word_sorting_mistake_verifier import (
    WordSortingMistakeVerifier,
)


class WordSortingMistake(Game):
    def __init__(self, word_count_range=[15, 25], mistake_rate=0.9, language="mixed"):
        super().__init__("Word Sorting Mistake", WordSortingMistakeVerifier)
        self.word_count_min = word_count_range[0]
        self.word_count_max = word_count_range[1]
        self.mistake_rate = mistake_rate
        self.language = language
        self.alphabet = list("abcdefghijklmnopqrstuvwxyz")
        self.english_words = self.load_words()
        self.letter_order = {letter: i for i, letter in enumerate(self.alphabet)}
        self.letter_order[""] = -1

    def load_words(self):
        current_dir = pathlib.Path(__file__).parent.resolve()
        words_path = current_dir / "words_alpha.txt"
        with open(words_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def list2str(self, words, error_word_map):
        true_words = [error_word_map[word] if word in error_word_map else word for word in words]
        if len(words) == 1:
            return f'"{true_words[0]}"'
        else:
            return "[" + " ? ".join([f'"{word}"' for word in true_words]) + "]"

    def position2str(self, position):
        position_dic = {
            0: "first",
            1: "second",
            2: "third",
            3: "fourth",
            4: "fifth",
            5: "sixth",
            6: "seventh",
            7: "eighth",
            8: "ninth",
            9: "tenth",
            10: "eleventh",
            11: "twelfth",
            12: "thirteenth",
            13: "fourteenth",
            14: "fifteenth",
            15: "sixteenth",
            16: "seventeenth",
            17: "eighteenth",
            18: "nineteenth",
            19: "twentieth",
        }
        return position_dic[position]

    def position2str_chinese(self, position):
        position_dic_chinese = {
            0: "第一个",
            1: "第二个",
            2: "第三个",
            3: "第四个",
            4: "第五个",
            5: "第六个",
            6: "第七个",
            7: "第八个",
            8: "第九个",
            9: "第十个",
            10: "第十一个",
            11: "第十二个",
            12: "第十三个",
            13: "第十四个",
            14: "第十五个",
            15: "第十六个",
            16: "第十七个",
            17: "第十八个",
            18: "第十九个",
            19: "第二十个",
        }
        return position_dic_chinese[position]

    def get_letter_order(self, word, position):
        if len(word) <= position:
            return -1
        return self.letter_order[word[position].lower()]

    def sort_list(self, sub_list, position, is_error=False):
        # order_dic = {}
        grouped = defaultdict(list)
        for word in sub_list:
            value = self.get_letter_order(word, position)
            grouped[value].append(word)
        sorting_state = []
        for value in sorted(grouped.keys()):
            sorting_state.append(grouped[value])
        if is_error:
            if len(sorting_state) > 1:
                if random.random() < 0.33:
                    exchange_position = random.sample(range(len(sorting_state)), 2)
                    sorting_state[exchange_position[0]], sorting_state[exchange_position[1]] = (
                        sorting_state[exchange_position[1]],
                        sorting_state[exchange_position[0]],
                    )
                elif random.random() < 0.66:
                    merge_position = random.sample(range(len(sorting_state)), 2)
                    sorting_state[merge_position[0]] = (
                        sorting_state[merge_position[0]] + sorting_state[merge_position[1]]
                    )
                    sorting_state.pop(merge_position[1])
                else:
                    insert_position = random.sample(range(len(sorting_state)), 2)
                    sorting_state.insert(insert_position[0], sorting_state[insert_position[1]])
            else:
                is_error = False
        return (sorting_state, is_error)

    def generate_sorting_process_with_mistake(self, words, language):
        thought = ""
        sorting_state = [words.copy()]
        is_mistake = random.random() < self.mistake_rate
        if is_mistake:
            mistake_step = random.randint(1, 10)
        else:
            mistake_step = None
        position = 0
        step_counter = 1
        error_word_map = {}
        while any((len(sublist) > 1 for sublist in sorting_state)):
            # position_to_check = position
            if position == 0:
                if language == "english":
                    thought = f"Thought {step_counter}:I should start by looking at the {self.position2str(position)} letter of the words in the list. The {self.position2str(position)} letter: "
                else:
                    thought = f"思考 {step_counter}:我应该先看列表中每个单词的{self.position2str_chinese(position)}字母。{self.position2str_chinese(position)}字母: "
                first_letters = {}
                if is_mistake and step_counter == mistake_step:
                    avail_words = [word for word in words if len(word) > position]
                    error_word = random.randint(0, len(avail_words) - 1)
                    original_word = avail_words[error_word]
                    replace_char = random.choice(
                        [word[position] for word in avail_words if word[position] != avail_words[error_word][position]]
                    )
                    replace_word = (
                        avail_words[error_word][:position] + replace_char + avail_words[error_word][position + 1 :]
                    )
                    error_word_map[replace_word] = avail_words[error_word]
                    words[words.index(original_word)] = replace_word
                for word in words:
                    if word in error_word_map:
                        real_word = error_word_map[word]
                        first_letter = word[position].lower() if len(word) > position else ""
                        position_value = self.get_letter_order(word, position) + 1
                        first_letters[word] = f'"{real_word}": "{first_letter}" ({position_value}). '
                        continue
                    first_letter = word[position].lower()
                    position_value = self.get_letter_order(word, position) + 1
                    first_letters[word] = f'"{word}": "{first_letter}" ({position_value}). '
                thought += "".join(first_letters.values())
                step_counter += 1
                if is_mistake and step_counter == mistake_step:
                    sorting_state, is_error = self.sort_list(words, position, is_error=True)
                    if not is_error:
                        is_mistake = False
                        mistake_step = None
                else:
                    sorting_state = self.sort_list(words, position)[0]
                if language == "english":
                    thought += (
                        f"Thought {step_counter}: We now have: "
                        + " < ".join([self.list2str(sublist, error_word_map) for sublist in sorting_state])
                        + "."
                    )
                else:
                    thought += (
                        f"思考 {step_counter}: 我们现在有: "
                        + " < ".join([self.list2str(sublist, error_word_map) for sublist in sorting_state])
                        + "。"
                    )
                step_counter += 1
            else:
                new_sorting_state = []
                for i, sublist in enumerate(sorting_state):
                    if len(sublist) == 1:
                        new_sorting_state.append(sublist)
                        continue
                    if language == "english":
                        thought += f"Thought {step_counter}: Now let's sort this subpart {self.list2str(sublist, error_word_map)} by looking at the {self.position2str(position)} letter. The {self.position2str(position)} letter: "
                    else:
                        thought += f"思考 {step_counter}: 现在让我们按{self.position2str_chinese(position)}字母排序这个子列表{self.list2str(sublist, error_word_map)}。{self.position2str_chinese(position)}字母: "
                    if is_mistake and step_counter == mistake_step:
                        try:
                            avail_words = [word for word in sublist if len(word) > position]
                            error_word = random.randint(0, len(avail_words) - 1)
                            original_word = avail_words[error_word]
                            replace_char = random.choice(
                                [char for char in self.alphabet if char != avail_words[error_word][position]]
                            )
                            replace_word = (
                                avail_words[error_word][:position]
                                + replace_char
                                + avail_words[error_word][position + 1 :]
                            )
                            error_word_map[replace_word] = avail_words[error_word]
                            sublist[sublist.index(original_word)] = replace_word
                        except Exception:
                            is_mistake = False
                            mistake_step = None
                    order_letters = {}
                    for word in sublist:
                        if word in error_word_map:
                            real_word = error_word_map[word]
                            order_letter = word[position].lower() if len(word) > position else ""
                            position_value = self.get_letter_order(word, position) + 1
                            order_letters[word] = f'"{real_word}": "{order_letter}" ({position_value}). '
                            continue
                        order_letter = word[position].lower() if len(word) > position else ""
                        position_value = self.get_letter_order(word, position) + 1
                        order_letters[word] = f'"{word}": "{order_letter}" ({position_value}). '
                    thought += "".join(order_letters.values())
                    step_counter += 1
                    if is_mistake and step_counter == mistake_step:
                        new_sorting_state_, is_error = self.sort_list(sublist, position, is_error=True)
                        new_sorting_state += new_sorting_state_
                        if not is_error:
                            is_mistake = False
                            mistake_step = None
                    else:
                        new_sorting_state += self.sort_list(sublist, position)[0]
                    if language == "english":
                        thought += (
                            f"Thought {step_counter}: We now have: "
                            + " < ".join(
                                [self.list2str(l, error_word_map) for l in new_sorting_state + sorting_state[i + 1 :]]
                            )
                            + "."
                        )
                    else:
                        thought += (
                            f"思考 {step_counter}: 我们现在有: "
                            + " < ".join(
                                [self.list2str(l, error_word_map) for l in new_sorting_state + sorting_state[i + 1 :]]
                            )
                            + "。"
                        )
                    step_counter += 1
                sorting_state = new_sorting_state
            position += 1
        if is_mistake:
            if step_counter < mistake_step:
                is_mistake = False
                mistake_step = None
        if language == "english":
            final_thought = (
                "I have now sorted all the words. The answer is "
                + "<".join([self.list2str(l, error_word_map) for l in sorting_state])
                + "."
            )
        else:
            final_thought = (
                "我已经排序了所有单词。答案是"
                + "<".join([self.list2str(l, error_word_map) for l in sorting_state])
                + "。"
            )
        thought += final_thought
        return (thought, is_mistake, mistake_step)

    def generate_problem(self):
        if self.language == "mixed":
            language = random.choice(["english", "chinese"])
        else:
            language = self.language
        word_count = random.randint(self.word_count_min, self.word_count_max)
        words = random.sample(self.english_words, word_count)
        thoughts, is_mistake, mistake_step = self.generate_sorting_process_with_mistake(words, language)
        if language == "english":
            prompt = random.choice(english_prompts).format(word_list=" ".join(words), think_process=thoughts)
        else:
            prompt = random.choice(chinese_prompts).format(word_list=" ".join(words), think_process=thoughts)
        return {
            "question": prompt,
            "answer": mistake_step,
            "words": words,
            "thoughts": thoughts,
            "is_mistake": is_mistake,
        }

    def extract_answer(self, test_answer):
        return self.verifier.extract_answer(test_answer)

    def verify(self, data, test_answer):
        return self.verifier.verify(data, test_answer)

    def generate(self, num_of_questions=100, max_attempts=100):
        outputs = []
        for i in range(num_of_questions):
            result = self.generate_problem()
            outputs.append(
                Data(
                    question=result["question"],
                    answer=result["answer"],
                    difficulty=1,
                    metadata={
                        "is_mistake": result["is_mistake"],
                        "thoughts": result["thoughts"],
                        "words": result["words"],
                        "answer": result["answer"],
                    },
                )
            )
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成带有排序错误的单词排序游戏数据")
    parser.add_argument("--num_of_data", type=int, default=1000, help="生成的题目数量")
    parser.add_argument(
        "--word_count_range", type=int, nargs=2, default=[15, 20], help="生成单词数量范围 [最小值, 最大值]"
    )
    parser.add_argument("--mistake_rate", type=float, default=0.9, help="生成错误步骤的概率")
    parser.add_argument("--language", type=str, default="mixed", help="语言类型")
    args = parser.parse_args()
    game = WordSortingMistake(word_count_range=args.word_count_range)
    data_list = game.generate(num_of_questions=args.num_of_data)
    current_dir = pathlib.Path(__file__).parent.parent.resolve()
    data_dir = current_dir / "data"
    data_dir.mkdir(exist_ok=True)
    wc_min, wc_max = args.word_count_range
    filename = f"data_wc{wc_min}-{wc_max}_n{args.num_of_data}.jsonl"
    file_path = data_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        for data in data_list:
            f.write(
                json.dumps(
                    {
                        "question": data.question,
                        "answer": data.answer,
                        "difficulty": data.difficulty,
                        "metadata": data.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
