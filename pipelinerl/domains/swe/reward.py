import difflib
import logging
import re
from typing import Dict, List, Tuple, TypedDict

from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

logger = logging.getLogger(__name__)


class FormatError(Exception):
    pass


class ChangeSimilarity(TypedDict):
    path: str
    pred_change: str
    oracle_change: str
    similarity: float


def parse_patch_for_gold_files(patch_text: str) -> List[str]:
    """Extract modified file paths from a unified diff patch."""
    if not patch_text:
        return []
    return re.findall(r'^--- a/(.+)$', patch_text, re.MULTILINE)


def generate_unified_diff(old_code: str, new_code: str, n_context: int = 3) -> str:
    diff = difflib.unified_diff(
        old_code.splitlines(),
        new_code.splitlines(),
        fromfile="old",
        tofile="new",
        lineterm="",
        n=n_context,
    )
    try:
        next(diff)
        next(diff)
        return "\n".join(diff)
    except StopIteration:
        return ""


def apply_edits_to_files(
    file_contents: Dict[str, str],
    edits: List[Dict],
    silent: bool = False,
) -> Dict[str, str]:
    new_content_dict = dict(file_contents)
    for edit in edits:
        file_path = edit.get('file_path', '')
        search_text = edit.get('search', '')
        replace_text = edit.get('replace', '')

        if not silent and search_text == replace_text:
            raise FormatError("Search and replace blocks are identical")

        if file_path not in new_content_dict:
            if not silent:
                raise FormatError(f"File {file_path} not found in file_contents")
            logger.warning("File %s not found in file_contents", file_path)
            continue

        current_content = new_content_dict[file_path]
        if search_text not in current_content:
            if not silent:
                raise FormatError(f"Search text not found in {file_path}: {search_text}")
            logger.warning("Search text not found in %s", file_path)
            continue

        new_content_dict[file_path] = current_content.replace(search_text, replace_text, 1)

    return new_content_dict


def get_normalized_patch(
    code_context: Dict[str, str],
    new_content_dict: Dict[str, str],
) -> Dict[str, str]:
    patch_dict = {}
    for path, new_content in new_content_dict.items():
        old_content = code_context.get(path, "")
        patch = generate_unified_diff(old_content, new_content)
        if patch:
            patch_dict[path] = patch
    return patch_dict


def get_filelevel_diff(patch_text: str) -> Dict[str, str]:
    try:
        patch = PatchSet(patch_text)
    except UnidiffParseError:
        return {}
    except Exception as e:
        logger.warning("Unexpected unidiff parsing error: %s", e)
        return {}

    result = {}
    for patchfile in patch:
        body = "\n".join(str(hunk).strip() for hunk in patchfile)
        result[patchfile.path] = body.strip()
    return result


def compute_change_similarities(
    pred_patch: Dict[str, str],
    oracle_patch: Dict[str, str],
) -> List[ChangeSimilarity]:
    all_file_paths = set(oracle_patch) | set(pred_patch)
    similarities = []
    for path in all_file_paths:
        pred_change = pred_patch.get(path, "")
        oracle_change = oracle_patch.get(path, "")
        if not oracle_change or not pred_change:
            change_similarity = 0.0
        else:
            change_similarity = difflib.SequenceMatcher(
                None, pred_change, oracle_change, autojunk=False
            ).ratio()
        similarities.append(ChangeSimilarity(
            path=path,
            pred_change=pred_change,
            oracle_change=oracle_change,
            similarity=change_similarity,
        ))
    return similarities


def calculate_precise_reward(
    file_contents: Dict[str, str],
    oracle_patch_text: str,
    predicted_edits: List[Dict],
) -> Tuple[float, Dict]:
    try:
        if not predicted_edits:
            raise FormatError("No valid search blocks found")

        oracle_patch = get_filelevel_diff(oracle_patch_text)
        pred_new_content = apply_edits_to_files(file_contents, predicted_edits)
        pred_patch = get_normalized_patch(file_contents, pred_new_content)
        similarities = compute_change_similarities(pred_patch, oracle_patch)

        if not similarities:
            assert not oracle_patch and not pred_patch
            return 1.0, {"similarities": []}

        reward = sum(s["similarity"] for s in similarities) / len(similarities)
        return reward, {
            "similarities": similarities,
            "num_files_changed": len(similarities),
            "oracle_files": list(oracle_patch.keys()),
            "predicted_files": list(pred_patch.keys()),
        }

    except FormatError as e:
        # logger.warning("Format error in reward calculation: %s", str(e))
        return 0.0, {"format_error": True, "error_message": str(e)}
    except Exception as e:
        logger.error("Unexpected error in reward calculation: %s", e)
        return 0.0, {"error": str(e)}
