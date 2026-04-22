import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful coding assistant that analyzes code and fixes bugs."

USER_PROMPT_TEMPLATE = (
    "Analyze the following code to find and fix bugs. Use this format:\n\n"
    "<think>\n"
    "[Your analysis process - be as detailed as you want until you're confident in your solution]\n"
    "</think>\n\n"
    "<solution>\n"
    "[Your SEARCH/REPLACE edits using this format:]\n\n"
    "### filename.py\n"
    "<<<<<<< SEARCH\n"
    "[exact code to find]\n"
    "=======\n"
    "[replacement code]\n"
    ">>>>>>> REPLACE\n"
    "</solution>\n\n"
    "IMPORTANT REQUIREMENTS:\n"
    "- Every SEARCH/REPLACE edit must use the exact format above\n"
    "- The SEARCH block must contain a contiguous chunk of lines that exist in the source code\n"
    "- PROPER INDENTATION IS CRITICAL - if you want to add '    print(x)', you must include all those spaces\n"
    "- Wrap each SEARCH/REPLACE edit in a code block\n"
    "- Use separate code blocks for multiple edits\n\n"
    "Example:\n"
    "```python\n"
    "### mathweb/flask/app.py\n"
    "<<<<<<< SEARCH\n"
    "from flask import Flask\n"
    "=======\n"
    "import math\n"
    "from flask import Flask\n"
    ">>>>>>> REPLACE\n"
    "```\n\n"
    "Here is the issue:\n"
    "--- BEGIN ISSUE ---\n"
    "{problem_statement}\n"
    "--- END ISSUE ---\n\n"
    "Below are the code files that may contain bugs:\n"
    "{file_contents}"
)


def build_messages(problem_statement: str, file_contents: Dict[str, str]) -> List[dict]:
    """Build the chat messages for a single-turn repair prompt."""
    formatted_files = "".join(
        f"### {path}\n```\n{content}\n```\n\n"
        for path, content in file_contents.items()
    )
    user_content = USER_PROMPT_TEMPLATE.format(
        problem_statement=problem_statement,
        file_contents=formatted_files,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_edits(completion: str) -> List[dict]:
    """
    Parse SEARCH/REPLACE blocks from a model completion.

    Each code block must start with '### filepath' and contain exactly one
    <<<<<<< SEARCH / ======= / >>>>>>> REPLACE triple.
    Returns a list of {'file_path', 'search', 'replace'} dicts.
    """
    edits = []
    code_blocks = _extract_code_blocks(completion)

    for block in code_blocks:
        edit = _parse_single_block(block)
        if edit is not None:
            edits.append(edit)

    return edits


def _extract_code_blocks(text: str) -> List[str]:
    blocks = []
    in_block = False
    current: List[str] = []
    for line in text.split('\n'):
        if line.strip().startswith('```'):
            if in_block:
                blocks.append('\n'.join(current))
                current = []
            in_block = not in_block
        elif in_block:
            current.append(line)
    return blocks


def _parse_single_block(block: str) -> dict | None:
    lines = block.split('\n')

    file_path = None
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('###'):
            file_path = line.strip()[3:].strip()
            start_idx = i + 1
            break

    if not file_path:
        return None

    search_start = search_end = replace_start = replace_end = None
    for i, line in enumerate(lines[start_idx:], start=start_idx):
        if '<<<<<<< SEARCH' in line:
            search_start = i + 1
        elif '=======' in line and search_start is not None and search_end is None:
            search_end = i
            replace_start = i + 1
        elif '>>>>>>> REPLACE' in line and replace_start is not None:
            replace_end = i
            break

    if None in (search_start, search_end, replace_start, replace_end):
        return None

    return {
        'file_path': file_path,
        'search': '\n'.join(lines[search_start:search_end]),
        'replace': '\n'.join(lines[replace_start:replace_end]),
    }
