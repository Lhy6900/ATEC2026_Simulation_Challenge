import importlib
import sys
import types


class _FakeBoxPushPlannerSolution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_action_spec(self):
        return {}

    def predicts(self, obs, current_score):
        return {"action": [[0.0]], "giveup": False}

    def reset(self, **kwargs):
        return None

    def get_debug_snapshot(self):
        return {}


class _FakeCrossPitBoxPolicy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def predicts(self, obs, current_score):
        return {"action": [[0.0]], "giveup": False}

    def reset(self, **kwargs):
        return None

    def get_debug_snapshot(self):
        return {}


def _load_solution_module(monkeypatch):
    fake_box_module = types.ModuleType("demo.boxpush_planner_solution")
    fake_box_module.BoxPushPlannerSolution = _FakeBoxPushPlannerSolution
    monkeypatch.setitem(sys.modules, "demo.boxpush_planner_solution", fake_box_module)

    fake_cross_module = types.ModuleType("demo.cross_pit_box_policy")
    fake_cross_module.CrossPitBoxPolicy = _FakeCrossPitBoxPolicy
    monkeypatch.setitem(sys.modules, "demo.cross_pit_box_policy", fake_cross_module)

    import demo.solution as solution_module

    return importlib.reload(solution_module)


def test_step_counter_is_enabled_by_default_every_50_steps(monkeypatch, capsys):
    monkeypatch.delenv("ATEC_DEBUG_STEP_INTERVAL", raising=False)
    solution_module = _load_solution_module(monkeypatch)
    solution = solution_module.AlgSolution()

    for _ in range(49):
        solution.predicts({}, 0.0)

    assert "[DEBUG] step=" not in capsys.readouterr().out

    solution.predicts({}, 0.0)

    assert "[DEBUG] step=50" in capsys.readouterr().out


def test_step_counter_interval_can_be_overridden(monkeypatch, capsys):
    monkeypatch.setenv("ATEC_DEBUG_STEP_INTERVAL", "3")
    solution_module = _load_solution_module(monkeypatch)
    solution = solution_module.AlgSolution()

    solution.predicts({}, 0.0)
    solution.predicts({}, 0.0)

    assert "[DEBUG] step=" not in capsys.readouterr().out

    solution.predicts({}, 0.0)

    assert "[DEBUG] step=3" in capsys.readouterr().out
