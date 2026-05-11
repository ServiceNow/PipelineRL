import logging
from typing import List

from datasets import load_dataset

logger = logging.getLogger(__name__)

DOMAIN = "mathv"


def _format_choices(choices) -> str:
    if not choices:
        return ""
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    return "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(choices))


def _first_image(item: dict):
    # MathVista ships both `image` (string path) and `decoded_image` (PIL Image);
    # geometry3k ships `images` (list of PIL Images). Prefer fields that hold the
    # actual decoded image, and skip string paths since downstream code expects PIL.
    # NOTE: geometry3k and MathVista testmini are single-image-per-item, so taking
    # [0] of `images` loses nothing here. If a multi-image dataset is added (e.g.
    # MMMU, MathVista full test), this needs to return a list and the rollout's
    # create_multimodal_message must emit multiple image_url content blocks.
    img = item.get("decoded_image")
    if img is not None and not isinstance(img, str):
        return img
    if "images" in item and item["images"]:
        first = item["images"][0]
        if not isinstance(first, str):
            return first
    img = item.get("image")
    if img is not None and not isinstance(img, str):
        return img
    return None


def process_geometry3k(dataset, dataset_name: str):
    """hiyouga/geometry3k schema: {images, problem, answer}. The `problem`
    field already contains the choices inline."""
    for item in dataset:
        image = _first_image(item)
        if image is None or "problem" not in item or "answer" not in item:
            continue
        try:
            yield {
                "dataset": dataset_name,
                "image": image,
                "question": item["problem"],
                "answer": str(item["answer"]).strip(),
            }
        except Exception as e:
            logger.error(f"Error processing geometry3k item: {e}")
            continue


def process_mathvista(dataset, dataset_name: str):
    """AI4Math/MathVista schema: {pid, question, choices, answer,
    question_type, decoded_image, ...}."""
    for item in dataset:
        image = _first_image(item)
        if image is None or "question" not in item or item.get("answer") is None:
            continue
        try:
            question = item["question"]
            choices_block = _format_choices(item.get("choices"))
            if choices_block:
                question = f"{question}\n\nChoices:\n{choices_block}"
            yield {
                "dataset": dataset_name,
                "image": image,
                "question": question,
                "answer": str(item["answer"]).strip(),
            }
        except Exception as e:
            logger.error(f"Error processing mathvista item: {e}")
            continue


def add_ids(dataset: list[dict]):
    for i, entry in enumerate(dataset):
        entry["id"] = i
        entry.setdefault("domain", DOMAIN)
    return dataset


def load_problems(dataset_names: List[str] | str | None) -> List[dict]:
    """Load math-visual datasets and return a list of standardized problems."""
    if dataset_names is None:
        return []

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    out: list[dict] = []

    if "geometry3k_train" in dataset_names:
        ds = load_dataset("hiyouga/geometry3k", split="train", trust_remote_code=True)
        out += add_ids(list(process_geometry3k(ds, "geometry3k_train")))

    if "geometry3k_test" in dataset_names:
        ds = load_dataset("hiyouga/geometry3k", split="test", trust_remote_code=True)
        out += add_ids(list(process_geometry3k(ds, "geometry3k_test")))

    if "geometry3k_val" in dataset_names:
        ds = load_dataset("hiyouga/geometry3k", split="validation", trust_remote_code=True)
        out += add_ids(list(process_geometry3k(ds, "geometry3k_val")))

    if "mathvista_testmini" in dataset_names:
        ds = load_dataset("AI4Math/MathVista", split="testmini", trust_remote_code=True)
        out += add_ids(list(process_mathvista(ds, "mathvista_testmini")))

    if not out:
        raise ValueError(f"No mathv datasets loaded from {dataset_names!r}")

    return out
