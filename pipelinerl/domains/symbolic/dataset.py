from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import sympy as sp

DOMAIN = "symbolic"
DEFAULT_VARIABLES = ("x",)

ProblemBuilder = Callable[[random.Random, "DatasetSpec"], dict]


def _random_poly(rng: random.Random, symbol: sp.Symbol, degree_range: tuple[int, int], terms_range: tuple[int, int]) -> sp.Expr:
    degree = rng.randint(*degree_range)
    terms = max(2, rng.randint(*terms_range))
    expr = 0
    for _ in range(terms):
        coeff = rng.randint(-6, 6)
        if coeff == 0:
            coeff = rng.choice([-3, -2, -1, 1, 2, 3])
        power = rng.randint(0, degree)
        expr += coeff * sp.Pow(symbol, power)
    return expr


def _ensure_non_constant(expr: sp.Expr, symbol: sp.Symbol, rng: random.Random) -> sp.Expr:
    if expr.free_symbols:
        return expr
    # Force dependence on the variable by adding a simple linear term.
    return expr + rng.randint(1, 5) * symbol


def _sympy_str(expr: sp.Expr) -> str:
    return sp.sstr(expr)


def _format_boxed(expr: sp.Expr | str) -> str:
    text = expr if isinstance(expr, str) else _sympy_str(expr)
    return f"\\boxed{{{text}}}"


def _ordinal(n: int) -> str:
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = suffixes.get(n % 10, "th")
    return f"{n}{suffix}"


def _derivative_problem(rng: random.Random, dataset: "DatasetSpec", *, trig: bool = False) -> dict:
    symbol = dataset.primary_symbol
    expr = _random_poly(rng, symbol, dataset.degree_range, dataset.terms_range)
    if trig:
        trig_fn = rng.choice([sp.sin, sp.cos, sp.tan])
        expr += rng.randint(1, 4) * trig_fn(symbol * rng.randint(1, 3))
    expr = _ensure_non_constant(expr, symbol, rng)
    order = 1 if dataset.max_derivative_order <= 1 else rng.randint(1, dataset.max_derivative_order)
    derivative = sp.simplify(sp.diff(expr, symbol, order))
    prompt = (
        f"Compute the {_ordinal(order)} derivative with respect to {symbol} of the expression:\n"
        f"{_sympy_str(expr)}\n"
        "Use SymPy syntax (e.g., x**2, sin(x), exp(x)). Return only the final expression inside \\boxed{ }."
    )
    return {
        "task": prompt,
        "answer": _format_boxed(derivative),
        "target": _sympy_str(derivative),
        "expression": _sympy_str(expr),
        "task_type": "derivative",
        "variables": dataset.variables,
    }


def _simplify_problem(rng: random.Random, dataset: "DatasetSpec") -> dict:
    symbol = dataset.primary_symbol
    pieces = [_random_poly(rng, symbol, dataset.degree_range, dataset.terms_range) for _ in range(3)]
    expr = pieces[0]
    if rng.random() < 0.6:
        expr = (pieces[0] + pieces[1]) * (symbol + rng.randint(-3, 3))
    expr += rng.choice([pieces[1], pieces[2], pieces[1] - pieces[2], pieces[2] - pieces[0]])
    expr = _ensure_non_constant(expr, symbol, rng)
    unsimplified = expr
    target = sp.simplify(sp.expand(expr))
    prompt = (
        "Simplify the following expression so that like terms are combined and factors are expanded when helpful:\n"
        f"{_sympy_str(unsimplified)}\n"
        f"Use only SymPy syntax with the variables {', '.join(dataset.variables)} and return the final result inside \\boxed{{}}."
    )
    return {
        "task": prompt,
        "answer": _format_boxed(target),
        "target": _sympy_str(target),
        "expression": _sympy_str(unsimplified),
        "task_type": "simplify",
        "variables": dataset.variables,
    }


BUILDERS: dict[str, ProblemBuilder] = {
    "simplify_poly": _simplify_problem,
    "differentiate_poly": _derivative_problem,
    "differentiate_trig": lambda rng, spec: _derivative_problem(rng, spec, trig=True),
}


@dataclass
class DatasetSpec:
    name: str
    mix: Sequence[str]
    degree_range: tuple[int, int]
    terms_range: tuple[int, int]
    max_derivative_order: int = 1
    default_examples: int = 512
    variables: Sequence[str] = DEFAULT_VARIABLES

    @property
    def primary_symbol(self) -> sp.Symbol:
        return sp.symbols(self.variables[0])


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "cas_mix_easy": DatasetSpec(
        name="cas_mix_easy",
        mix=("simplify_poly", "differentiate_poly"),
        degree_range=(1, 3),
        terms_range=(2, 4),
        max_derivative_order=1,
        default_examples=512,
    ),
    "cas_mix_trig": DatasetSpec(
        name="cas_mix_trig",
        mix=("simplify_poly", "differentiate_poly", "differentiate_trig"),
        degree_range=(2, 4),
        terms_range=(2, 5),
        max_derivative_order=2,
        default_examples=512,
    ),
}


def _resolve_spec(dataset_name: str) -> DatasetSpec:
    base_name = dataset_name.split("@", 1)[0].strip()
    if base_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown symbolic dataset '{dataset_name}'. Available datasets: {', '.join(sorted(DATASET_REGISTRY))}"
        )
    return DATASET_REGISTRY[base_name]


def _normalize_mix(raw_mix: Sequence[str] | None, default: Sequence[str]) -> Sequence[str]:
    if not raw_mix:
        return default
    normalized: list[str] = []
    for item in raw_mix:
        key = item.strip()
        if key not in BUILDERS:
            raise ValueError(f"Unknown symbolic task '{item}'. Valid tasks: {', '.join(sorted(BUILDERS))}")
        normalized.append(key)
    return normalized or default


def _extract_split(raw_name: str) -> str | None:
    if "@" not in raw_name:
        return None
    split = raw_name.split("@", 1)[1].strip()
    return split or None


def _sanitize_token(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def load_problems(
    dataset_names: Iterable[str] | str | None,
    *,
    seed: int | None = None,
    max_examples_per_split: int | None = None,
    task_mix: Sequence[str] | None = None,
    degree_range: tuple[int, int] | None = None,
    terms_range: tuple[int, int] | None = None,
    max_derivative_order: int | None = None,
    **_: dict,
) -> list[dict]:
    if not dataset_names:
        return []
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    problems: list[dict] = []
    base_seed = seed or 0
    for raw_name in dataset_names:
        spec = _resolve_spec(raw_name)
        dataset_seed = base_seed ^ (zlib.crc32(raw_name.encode()) & 0xFFFFFFFF)
        rng = random.Random(dataset_seed)
        limit_hint = max_examples_per_split or spec.default_examples
        limit = max(1, limit_hint)
        mix = _normalize_mix(task_mix, spec.mix)
        degree = degree_range or spec.degree_range
        terms = terms_range or spec.terms_range
        derivative_order = max_derivative_order or spec.max_derivative_order
        custom_spec = DatasetSpec(
            name=spec.name,
            mix=mix,
            degree_range=degree,
            terms_range=terms,
            max_derivative_order=derivative_order,
            default_examples=limit,
            variables=tuple(spec.variables),
        )
        builder_choices = tuple(custom_spec.mix)
        split_suffix = _extract_split(raw_name)
        dataset_label = f"{DOMAIN}/{spec.name}"
        if split_suffix:
            dataset_label = f"{dataset_label}@{split_suffix}"
        id_prefix = f"{spec.name}-{_sanitize_token(split_suffix) if split_suffix else 'default'}"
        for idx in range(limit):
            builder_key = rng.choice(builder_choices)
            builder = BUILDERS[builder_key]
            problem = builder(rng, custom_spec)
            problem.update(
                {
                    "domain": DOMAIN,
                    "dataset": dataset_label,
                    "id": f"{id_prefix}-{idx}",
                }
            )
            problems.append(problem)
    return problems
