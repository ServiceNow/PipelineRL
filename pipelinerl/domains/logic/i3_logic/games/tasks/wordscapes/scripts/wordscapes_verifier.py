import re

from .....base.verifier import THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START, Verifier

debug_mode = False


class WordscapesVerifier(Verifier):
    def verify(self, data, test_solution: str):
        try:
            extracted_answer = self.extract_answer(test_solution)
            if not extracted_answer:
                return False
            grid = data.metadata["grid"]
            across_words = data.metadata["across_words"]
            down_words = data.metadata["down_words"]
            if len(extracted_answer) != len(grid):
                return False
            for i in range(len(grid)):
                if len(extracted_answer[i]) != len(grid[i]):
                    return False
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    if grid[i][j] == "0" and extracted_answer[i][j].strip():
                        return False
                    if grid[i][j] == "X" and (not extracted_answer[i][j].strip()):
                        return False
            for word in across_words:
                found = False
                for i in range(len(extracted_answer)):
                    row_str = "".join(extracted_answer[i]).replace(" ", "").lower()
                    if word.lower() in row_str:
                        found = True
                        break
                if not found and word:
                    return 0
            for word in down_words:
                found = False
                for j in range(len(extracted_answer[0])):
                    col = []
                    for i in range(len(extracted_answer)):
                        if j < len(extracted_answer[i]):
                            col.append(extracted_answer[i][j])
                    col_str = "".join(col).replace(" ", "").lower()
                    if word.lower() in col_str:
                        found = True
                        break
                if not found and word:
                    return False
            return True
        except Exception:
            return False

    def extract_answer(self, test_solution: str):
        try:
            if THOUGHT_DELIMITER_START in test_solution and THOUGHT_DELIMITER_END in test_solution:
                thought_end_pos = test_solution.rfind(THOUGHT_DELIMITER_END)
                if thought_end_pos >= 0:
                    test_solution = test_solution[thought_end_pos + len(THOUGHT_DELIMITER_END) :]
            grid_pattern = re.search("\\[\\s*\\[(?:\\s*\\[)?(.+?)(?:\\]\\s*)?\\]\\s*\\]", test_solution, re.DOTALL)
            if not grid_pattern:
                return None
            grid_text = grid_pattern.group(1)
            rows = []
            split_rows = re.split("\\],\\s*\\[", grid_text)
            for row_text in split_rows:
                row_text = row_text.strip().strip("[],")
                chars = []
                char_matches = re.findall("\\\"([^\\\"]*)\\\"|\\'([^\\']*)\\'|([^,\\s]+)", row_text)
                for match in char_matches:
                    char = next((x for x in match if x), "")
                    if char == "0" or char == "":
                        char = " "
                    chars.append(char)
                if chars:
                    rows.append(chars)
            if not rows or not all(rows):
                return None
            return rows
        except Exception:
            return None
