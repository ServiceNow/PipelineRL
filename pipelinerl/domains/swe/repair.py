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
    "```\n"
    "### filename.py\n"
    "<<<<<<< SEARCH\n"
    "[exact code to find]\n"
    "=======\n"
    "[replacement code]\n"
    ">>>>>>> REPLACE\n"
    "```\n"
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

    Each block is a '### filepath' line followed by a
    <<<<<<< SEARCH / ======= / >>>>>>> REPLACE triple. Triple-backtick code
    fences around the block are accepted but not required.
    Returns a list of {'file_path', 'search', 'replace'} dicts.
    """
    edits: List[dict] = []
    lines = completion.split('\n')
    n = len(lines)
    i = 0
    while i < n:
        if '<<<<<<< SEARCH' not in lines[i]:
            i += 1
            continue

        # Walk back to the most recent '### filepath' line, but don't cross a
        # previous '>>>>>>> REPLACE' marker (that path belongs to an earlier edit).
        file_path = None
        for j in range(i - 1, -1, -1):
            if '>>>>>>> REPLACE' in lines[j]:
                break
            stripped = lines[j].strip()
            if stripped.startswith('###'):
                file_path = stripped[3:].strip()
                break
        if not file_path:
            i += 1
            continue

        search_start = i + 1
        sep = replace_end = None
        for k in range(search_start, n):
            if sep is None and '=======' in lines[k]:
                sep = k
            elif sep is not None and '>>>>>>> REPLACE' in lines[k]:
                replace_end = k
                break

        if sep is None or replace_end is None:
            i += 1
            continue

        edits.append({
            'file_path': file_path,
            'search': '\n'.join(lines[search_start:sep]),
            'replace': '\n'.join(lines[sep + 1:replace_end]),
        })
        i = replace_end + 1
    return edits
