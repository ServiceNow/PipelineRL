from __future__ import annotations


def select_worker_for_cube(
    *,
    cube_id: str,
    candidate_indices: list[int],
    active_rollouts_by_worker: list[int],
    worker_max_rollouts: int,
    current_cube_by_worker: list[str | None],
) -> int | None:
    available = [
        idx
        for idx in candidate_indices
        if active_rollouts_by_worker[idx] < worker_max_rollouts
    ]
    if not available:
        return None

    def score(idx: int) -> tuple[int, int, int]:
        current_cube = current_cube_by_worker[idx]
        if current_cube == cube_id:
            affinity = 0
        elif current_cube is None:
            affinity = 1
        else:
            affinity = 2
        return (affinity, active_rollouts_by_worker[idx], idx)

    return min(available, key=score)
