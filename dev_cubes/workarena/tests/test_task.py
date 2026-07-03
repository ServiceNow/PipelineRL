"""Tests for workarena_cube.task module."""

import browsergym.workarena as wa

from workarena_cube.task import _load_task_class

_L2_CLASS_PATH = "browsergym.workarena.tasks.compositional.PriorityFilterProblemsAndMarkDuplicatesSmallTaskL2"
_L3_CLASS_PATH = "browsergym.workarena.tasks.compositional.PriorityFilterProblemsAndMarkDuplicatesSmallTaskL3"


def test_load_task_class_l2() -> None:
    cls = _load_task_class(_L2_CLASS_PATH)
    assert cls in wa.ALL_WORKARENA_TASKS
    assert cls.__name__ == "PriorityFilterProblemsAndMarkDuplicatesSmallTaskL2"


def test_load_task_class_l3() -> None:
    cls = _load_task_class(_L3_CLASS_PATH)
    assert cls in wa.ALL_WORKARENA_TASKS
    assert cls.__name__ == "PriorityFilterProblemsAndMarkDuplicatesSmallTaskL3"
