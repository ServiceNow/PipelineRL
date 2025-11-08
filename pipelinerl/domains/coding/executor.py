from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass
class ExecutionResult:
    passed: int
    total: int
    compile_error: str | None = None
    runtime_error: str | None = None
    timeout: bool = False
    details: list[dict[str, Any]] | None = None
    stdout: str = ""
    stderr: str = ""
    wall_time: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total": self.total,
            "compile_error": self.compile_error,
            "runtime_error": self.runtime_error,
            "timeout": self.timeout,
            "details": self.details or [],
            "stdout": self.stdout,
            "stderr": self.stderr,
            "wall_time": self.wall_time,
        }


def _runner_script() -> str:
    return textwrap.dedent(
        r"""
        import argparse
        import ast
        import importlib.util
        import json
        import math
        import resource
        import signal
        import subprocess
        import sys
        import traceback


        def _literal_eval(token: str):
            token = token.strip()
            if not token:
                return None
            lowered = token.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            if lowered in {"null", "none"}:
                return None
            try:
                return ast.literal_eval(token)
            except Exception:
                return token


        def _parse_inputs(raw: str) -> list:
            raw = (raw or "").strip()
            if not raw:
                return []
            try:
                value = _literal_eval(raw)
                return value if isinstance(value, list) else [value]
            except Exception:
                pass
            args = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    args.append(_literal_eval(line))
                except Exception:
                    args.append(line)
            return args


        def _values_equal(a, b) -> bool:
            if isinstance(a, float) and isinstance(b, float):
                return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
            return a == b


        def _load_submission(path: str):
            spec = importlib.util.spec_from_file_location("submission", path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module


        def _call_target(module, entry_point: str | None, args: list):
            solution_cls = getattr(module, "Solution", None)
            if solution_cls is not None:
                method_name = entry_point
                instance = solution_cls()
                if method_name and hasattr(instance, method_name):
                    return getattr(instance, method_name)(*args)
                for attr in dir(instance):
                    if attr.startswith("_"):
                        continue
                    method = getattr(instance, attr)
                    if callable(method):
                        return method(*args)
            if entry_point and hasattr(module, entry_point):
                candidate = getattr(module, entry_point)
                if callable(candidate):
                    return candidate(*args)
            if hasattr(module, "solve"):
                solve_fn = getattr(module, "solve")
                if callable(solve_fn):
                    return solve_fn(*args)
            for attr in dir(module):
                if attr.startswith("_"):
                    continue
                candidate = getattr(module, attr)
                if callable(candidate):
                    return candidate(*args)
            raise AttributeError("entry_point_not_found")


        def _set_limits(mem_bytes: int, cpu_seconds: int):
            if mem_bytes > 0:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                resource.setrlimit(resource.RLIMIT_DATA, (mem_bytes, mem_bytes))
            if cpu_seconds > 0:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            resource.setrlimit(resource.RLIMIT_STACK, (256 * 1024 * 1024, 256 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))


        def _run_stdin_case(solution_path: str, case: dict, per_test_timeout: float):
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-B", solution_path],
                    input=case.get("input", ""),
                    capture_output=True,
                    text=True,
                    timeout=case.get("timeout", per_test_timeout),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return {"id": case["id"], "passed": False, "error": "timeout", "type": "stdin"}
            expected = (case.get("output") or "").strip()
            actual = (proc.stdout or "").strip()
            passed = actual == expected
            error = None
            if not passed and proc.returncode != 0:
                error = proc.stderr.strip() or f"exit_code={proc.returncode}"
            return {"id": case["id"], "passed": passed, "error": error, "type": "stdin"}


        def _run_functional_case(module, case: dict, entry_point: str | None):
            try:
                args = _parse_inputs(case.get("input", ""))
                expected = _literal_eval(case.get("output", ""))
                result = _call_target(module, entry_point, args)
            except AttributeError:
                return {"id": case["id"], "passed": False, "error": "entry_point_not_found", "type": "functional"}
            except Exception as exc:
                return {"id": case["id"], "passed": False, "error": f"runtime_error:{exc}", "type": "functional"}
            passed = _values_equal(result, expected)
            return {"id": case["id"], "passed": passed, "error": None if passed else "wrong_answer", "type": "functional"}


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--solution", required=True)
            parser.add_argument("--tests", required=True)
            parser.add_argument("--mem-bytes", type=int, default=512 * 1024 * 1024)
            parser.add_argument("--cpu-seconds", type=int, default=10)
            parser.add_argument("--per-test", type=float, default=4.0)
            args = parser.parse_args()

            _set_limits(args.mem_bytes, args.cpu_seconds)
            signal.alarm(args.cpu_seconds + 1)

            with open(args.tests, "r", encoding="utf-8") as f:
                payload = json.load(f)

            entry_point = payload.get("entry_point")
            cases = payload.get("tests", [])

            summary = {
                "passed": 0,
                "total": len(cases),
                "compile_error": None,
                "runtime_error": None,
                "timeout": False,
                "results": [],
            }

            try:
                module = _load_submission(args.solution)
            except Exception:
                summary["compile_error"] = traceback.format_exc(limit=4)
                print(json.dumps(summary))
                return

            for case in cases:
                case_type = case.get("type", "functional")
                if case_type == "stdin":
                    result = _run_stdin_case(args.solution, case, args.per_test)
                else:
                    result = _run_functional_case(module, case, entry_point)
                if result["passed"]:
                    summary["passed"] += 1
                elif result.get("error") == "runtime_error":
                    summary["runtime_error"] = result["error"]
                elif result.get("error") == "timeout":
                    summary["timeout"] = True
                summary["results"].append(result)

            print(json.dumps(summary))


        if __name__ == "__main__":
            main()
        """
    )


def run_coding_submission(
    code: str,
    tests: Sequence[dict],
    *,
    entry_point: str | None,
    time_limit_s: float = 10.0,
    per_test_timeout: float = 4.0,
    memory_limit_bytes: int = 512 * 1024 * 1024,
) -> ExecutionResult:
    """
    Execute a candidate solution inside an isolated Python subprocess.
    """
    start = time.time()
    total_tests = len(tests)
    if total_tests == 0:
        return ExecutionResult(passed=0, total=0, wall_time=0.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        submission_path = tmp_path / "submission.py"
        tests_path = tmp_path / "tests.json"
        runner_path = tmp_path / "runner.py"

        submission_path.write_text(code, encoding="utf-8")
        tests_payload = {
            "entry_point": entry_point,
            "tests": list(tests),
        }
        tests_path.write_text(json.dumps(tests_payload), encoding="utf-8")
        runner_path.write_text(_runner_script(), encoding="utf-8")

        env = {
            "PYTHONPATH": str(tmp_path),
            "PATH": os.getenv("PATH", ""),
        }

        cmd = [
            sys.executable,
            "-I",
            "-B",
            str(runner_path),
            "--solution",
            str(submission_path),
            "--tests",
            str(tests_path),
            "--mem-bytes",
            str(memory_limit_bytes),
            "--cpu-seconds",
            str(int(max(1, time_limit_s))),
            "--per-test",
            str(per_test_timeout),
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=time_limit_s,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            wall = time.time() - start
            return ExecutionResult(
                passed=0,
                total=total_tests,
                timeout=True,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + "\n<timeout>",
                wall_time=wall,
            )

        wall = time.time() - start
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        try:
            summary = json.loads(stdout.strip() or "{}")
        except json.JSONDecodeError:
            return ExecutionResult(
                passed=0,
                total=total_tests,
                runtime_error="runner_parse_failure",
                stdout=stdout,
                stderr=stderr,
                wall_time=wall,
            )

        return ExecutionResult(
            passed=int(summary.get("passed", 0)),
            total=int(summary.get("total", total_tests)),
            compile_error=summary.get("compile_error"),
            runtime_error=summary.get("runtime_error"),
            timeout=bool(summary.get("timeout", False)),
            details=summary.get("results"),
            stdout=stdout,
            stderr=stderr,
            wall_time=wall,
        )
