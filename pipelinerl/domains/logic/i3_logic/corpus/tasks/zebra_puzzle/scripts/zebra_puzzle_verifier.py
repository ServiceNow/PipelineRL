import json

from .....base import Data, Verifier


def _verifier(answer_str, user_response_str, type_safe=True):
    try:
        user_response_str = user_response_str.split("```json")[-1].split("```")[0].strip()
    except Exception:
        return False
    try:
        answer_obj = json.loads(answer_str)
    except json.JSONDecodeError:
        return answer_str == user_response_str
    else:
        if isinstance(answer_obj, dict):
            try:
                response_obj = json.loads(user_response_str)
            except json.JSONDecodeError:
                try:
                    response_obj = eval(user_response_str)
                except Exception:
                    return False
            if not isinstance(response_obj, dict):
                return False
            return _deep_compare(answer_obj, response_obj, type_safe)
        else:
            return answer_str == user_response_str


def _deep_compare(a, b, type_safe):
    if type_safe and type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return False
        return all((_deep_compare(a[k], b[k], type_safe) for k in a))
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return False
        return all((_deep_compare(x, y, type_safe) for x, y in zip(a, b)))
    else:
        return _compare_values(a, b, type_safe)


def _compare_values(a, b, type_safe):
    if type_safe:
        return a == b
    else:
        converted_a = _try_convert_number(a)
        converted_b = _try_convert_number(b)
        return converted_a == converted_b


def _try_convert_number(value):
    if isinstance(value, str):
        try:
            return int(value)
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            pass
    return value


class ZebraPuzzleVerifier(Verifier):
    def verify(self, data: Data, test_solution: str):
        extracted_answer = self.extract_answer(test_solution)
        ground_truth = data.answer
        try:
            correct = _verifier(answer_str=ground_truth, user_response_str=extracted_answer, type_safe=True)
        except Exception:
            correct = False
        if correct:
            acc_score = 1.0
        else:
            acc_score = 0
        return acc_score

    def extract_answer(self, test_solution: str):
        return test_solution
