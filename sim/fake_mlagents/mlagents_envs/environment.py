import json
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from thething_sim import TheThingWorld  # noqa: E402

from .base_env import BehaviorSpec, _Steps  # noqa: E402
from .side_channel.environment_parameters_channel import EnvironmentParametersChannel  # noqa: E402
from .side_channel.stats_side_channel import StatsSideChannel  # noqa: E402

BEHAVIOR = "TheThing?team=0"


class UnityEnvironment:
    """mlagents_envs.environment.UnityEnvironment 대체.
    SIM_SEED, SIM_GAME_LOG 환경변수로 시드와 게임 결과 로그 경로를 지정."""

    def __init__(self, file_name=None, side_channels=(), **kwargs):
        self.param_channel = next((c for c in side_channels if isinstance(c, EnvironmentParametersChannel)), None)
        self.stats_channel = next((c for c in side_channels if isinstance(c, StatsSideChannel)), None)
        self.game_log = []
        self.log_path = os.environ.get("SIM_GAME_LOG")
        self._logged = 0
        self.world = TheThingWorld(
            seed=int(os.environ.get("SIM_SEED", "0")),
            stats_sink=self.stats_channel.stats if self.stats_channel else None,
            game_log=self.game_log,
        )
        self.behavior_specs = {BEHAVIOR: BehaviorSpec(120, (9, 9))}
        self._actions = {}
        self._last = []

    def _sync(self):
        if self.param_channel is not None:
            self.world.env_params = dict(self.param_channel.params)
        if self.stats_channel is not None:
            self.world.stats_sink = self.stats_channel.stats
        if self.log_path and len(self.game_log) > self._logged:
            with open(self.log_path, "a", encoding="utf-8") as f:
                for g in self.game_log[self._logged:]:
                    f.write(json.dumps(g, ensure_ascii=False) + "\n")
            self._logged = len(self.game_log)

    def reset(self):
        self._sync()
        self.world.force_reset()
        self._last = self.world.first_exchange()
        if not self._last:
            self._last = self.world.advance({})
        self._actions = {}
        self._sync()

    def get_steps(self, behavior_name):
        dec = [(a, o, r) for a, o, r, d in self._last if not d]
        term = [(a, o, r) for a, o, r, d in self._last if d]
        self._dec_ids = [a for a, _, _ in dec]
        return _Steps(dec), _Steps(term)

    def set_actions(self, behavior_name, action_tuple):
        disc = action_tuple.discrete
        self._actions = {aid: disc[i] for i, aid in enumerate(self._dec_ids)}

    def step(self):
        self._sync()
        self._last = self.world.advance(self._actions)
        self._actions = {}
        self._sync()

    def close(self):
        self._sync()
