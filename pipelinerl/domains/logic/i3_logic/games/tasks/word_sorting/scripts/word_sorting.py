import argparse
import json
import pathlib
import random
from collections import defaultdict

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.word_sorting.scripts.word_sorting_prompt import chinese_prompts, english_prompts
from .....games.tasks.word_sorting.scripts.word_sorting_verifier import WordSortingVerifier


class WordSorting(Game):
    def __init__(self, front_letters_range=[1, 3], word_count_range=[15, 25]):
        super().__init__("Word Sorting", WordSortingVerifier)
        self.front_letters_min = front_letters_range[0]
        self.front_letters_max = front_letters_range[1]
        self.word_count_min = word_count_range[0]
        self.word_count_max = word_count_range[1]
        self.english_alphabet = list("abcdefghijklmnopqrstuvwxyz")
        self.english_words = self.load_words()
        self.words_by_first_letter = self.group_words_by_first_letter()

    def load_words(self):
        current_dir = pathlib.Path(__file__).parent.resolve()
        words_path = current_dir / "words_alpha.txt"
        with open(words_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def group_words_by_first_letter(self):
        grouped = defaultdict(list)
        for word in self.english_words:
            if word and len(word) > 0:
                first_letter = word[0].lower()
                grouped[first_letter].append(word)
        return grouped

    def create_new_alphabet(self):
        front_letters_count = random.randint(self.front_letters_min, self.front_letters_max)
        all_letters = list(self.english_alphabet)
        front_letters = random.sample(all_letters, front_letters_count)
        remaining_letters = [letter for letter in all_letters if letter not in front_letters]
        new_alphabet = front_letters + remaining_letters
        return (new_alphabet, front_letters)

    def sort_words(self, words, alphabet):
        letter_order = {letter: i for i, letter in enumerate(alphabet)}

        def get_word_key(word):
            return [letter_order.get(c, len(alphabet)) for c in word.lower()]

        return sorted(words, key=get_word_key)

    def select_words(self, count):
        selected_words = []
        available_groups = [letter for letter in self.words_by_first_letter if self.words_by_first_letter[letter]]
        words_by_first_letter_copy = {}
        for letter in self.words_by_first_letter:
            words_by_first_letter_copy[letter] = self.words_by_first_letter[letter].copy()
        while len(selected_words) < count and available_groups:
            group = random.choice(available_groups)
            word = random.choice(words_by_first_letter_copy[group])
            words_by_first_letter_copy[group].remove(word)
            selected_words.append(word)
        random.shuffle(selected_words)
        return selected_words

    def generate_problem(self):
        new_alphabet, front_letters = self.create_new_alphabet()
        word_count = random.randint(self.word_count_min, self.word_count_max)
        selected_words = self.select_words(word_count)
        sorted_words = self.sort_words(selected_words, new_alphabet)
        selected_words_str = ",".join(selected_words)
        order = random.choice([1, -1])
        order_str = {"english": {1: "increasing", -1: "decreasing"}, "chinese": {1: "升序", -1: "降序"}}
        prompt_dic = {"english": english_prompts, "chinese": chinese_prompts}
        language = random.choice(["english", "chinese"])
        if language == "english":
            front_letters_str = ",".join(front_letters[:-1]) + " and " + front_letters[-1]
        else:
            front_letters_str = "，".join(front_letters[:-1]) + "和" + front_letters[-1]
        question = random.choice(prompt_dic[language]).format(
            front_letters=front_letters_str, order=order_str[language][order], words=selected_words_str
        )
        if order == 1:
            answer = ", ".join(sorted_words)
        else:
            answer = ", ".join(sorted_words[::-1])
        return {
            "question": question,
            "answer": answer,
            "words": selected_words,
            "sorted_words": sorted_words,
            "new_alphabet": new_alphabet,
            "front_letters": front_letters,
            "order": order,
        }

    def verify(self, data, response):
        return self.verifier.verify(data, response)

    def extract_answer(self, response):
        return self.verifier.extract_answer(response)

    def generate(self, num_of_questions=100, max_attempts=100):
        outputs = []
        for i in range(num_of_questions):
            result = game.generate_problem()
            outputs.append(
                Data(
                    question=result["question"],
                    answer=result["answer"],
                    difficulty=1,
                    metadata={
                        "words": result["words"],
                        "sorted_words": result["sorted_words"],
                        "front_letters": ",".join(result["front_letters"]),
                        "order": result["order"],
                    },
                )
            )
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成单词排序游戏数据")
    parser.add_argument("--num_of_data", type=int, default=1000, help="生成的题目数量")
    parser.add_argument(
        "--front_letters_range", type=int, nargs=2, default=[2, 5], help="前置字母数量范围 [最小值, 最大值]"
    )
    parser.add_argument(
        "--word_count_range", type=int, nargs=2, default=[15, 25], help="生成单词数量范围 [最小值, 最大值]"
    )
    args = parser.parse_args()
    game = WordSorting(front_letters_range=args.front_letters_range, word_count_range=args.word_count_range)
    data_list = game.generate(num_of_questions=args.num_of_data)
    current_dir = pathlib.Path(__file__).parent.parent.resolve()
    data_dir = current_dir / "data"
    data_dir.mkdir(exist_ok=True)
    fl_min, fl_max = args.front_letters_range
    wc_min, wc_max = args.word_count_range
    filename = f"data_fl{fl_min}-{fl_max}_wc{wc_min}-{wc_max}_n{args.num_of_data}.jsonl"
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
