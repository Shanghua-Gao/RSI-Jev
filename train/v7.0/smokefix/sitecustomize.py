# a111-mix SMOKEFIX (2026-10-13): the frozen eval_smoke 1 (a111-mix pre-registration) keeps 1 question per target, so
# rsijev.evaluate._base_rate (majority class within a label space) is 1.0 and rsijev.metrics.decision_score divides by zero.
# On the train job's PYTHONPATH only: wrap decision_score so a choice score with base_rate >= 1 is NaN (undefined); all else unchanged.
# Training, the spec and the code dir are not touched. The full in-job eval is the 1-GPU guard job (eval_smoke null).
import importlib.abc, importlib.util, sys


class _SmokeFix(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "rsijev.metrics":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        run = spec.loader.exec_module

        def exec_module(mod):
            run(mod)
            f = mod.decision_score

            def decision_score(mode, **kw):
                b = kw.get("base_rate")
                if mode == "choice" and b is not None and b >= 1.0:
                    print(f"SMOKEFIX decision_score choice base_rate {b} -> nan", flush=True)
                    return float("nan")
                return f(mode, **kw)
            mod.decision_score = decision_score
            print(f"SMOKEFIX wrapped {mod.__file__}", flush=True)
        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _SmokeFix())
