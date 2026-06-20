import pytest


def test_resume_state_sync_advances_curriculum_from_loaded_iteration():
    from scripts.rsl_rl.resume_state import sync_resume_training_state
    from atec_rl_lab.train.cross_pit_box.mdp.curriculums import cross_pit_success_x_value

    class _CurriculumManager:
        def __init__(self, env):
            self.env = env
            self.calls = 0

        def compute(self, env_ids=None):
            self.calls += 1
            self.env.cross_pit_success_x = cross_pit_success_x_value(
                self.env,
                start_x=1.85,
                final_x=2.55,
                start_iteration=1000,
                end_iteration=6000,
                steps_per_iteration=32,
            )
            self.env.cross_pit_curriculum_progress = (self.env.cross_pit_success_x - 1.85) / (2.55 - 1.85)

        def reset(self, env_ids=None):
            return {
                "Curriculum/success_x/target_x": self.env.cross_pit_success_x,
                "Curriculum/success_x/progress": self.env.cross_pit_curriculum_progress,
            }

    class _UnwrappedEnv:
        common_step_counter = 0
        num_envs = 8192

        def __init__(self):
            self.curriculum_manager = _CurriculumManager(self)
            self.extras = {"log": {"Curriculum/success_x/target_x": 1.85}}

    class _WrappedEnv:
        def __init__(self):
            self.unwrapped = _UnwrappedEnv()
            self.num_envs = self.unwrapped.num_envs

    class _Runner:
        current_learning_iteration = 2800
        num_steps_per_env = 32
        gpu_world_size = 4
        tot_timesteps = 0

    env = _WrappedEnv()
    runner = _Runner()

    state = sync_resume_training_state(env, runner)

    assert state.iteration == 2800
    assert state.common_step_counter == 2800 * 32
    assert env.unwrapped.common_step_counter == 2800 * 32
    assert env.unwrapped.cross_pit_success_x == pytest.approx(2.102)
    assert env.unwrapped.curriculum_manager.calls == 1
    assert env.unwrapped.extras["log"]["Curriculum/success_x/target_x"] == pytest.approx(2.102)
    assert env.unwrapped.extras["log"]["Curriculum/success_x/progress"] == pytest.approx(0.36)
    assert state.total_timesteps == 2800 * 32 * 8192 * 4
    assert runner.tot_timesteps == state.total_timesteps


def test_resume_state_sync_is_noop_for_fresh_training():
    from scripts.rsl_rl.resume_state import sync_resume_training_state

    class _Env:
        common_step_counter = 0

    class _Runner:
        current_learning_iteration = 0
        num_steps_per_env = 32
        tot_timesteps = 0

    assert sync_resume_training_state(_Env(), _Runner()) is None
