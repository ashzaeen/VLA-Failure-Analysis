from __future__ import annotations

try:
    from libero.libero import benchmark
except Exception:  # pragma: no cover
    benchmark = None


class LiberoTaskSuiteLoader:
    def get_suite(self, task_suite_name: str):
        if benchmark is None:
            raise ImportError('LIBERO benchmark is not available in this environment.')
        benchmark_dict = benchmark.get_benchmark_dict()
        return benchmark_dict[task_suite_name]()
