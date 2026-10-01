import numpy as np


class ActionTuple:
    def __init__(self, continuous=None, discrete=None):
        self.continuous = continuous
        self.discrete = np.asarray(discrete) if discrete is not None else None


class _Step:
    def __init__(self, obs, reward):
        self.obs = [obs]
        self.reward = reward


class _Steps:
    """DecisionSteps / TerminalSteps 최소 구현"""

    def __init__(self, items):
        self._items = {int(aid): _Step(obs, r) for aid, obs, r in items}
        self.agent_id = np.array([int(aid) for aid, _, _ in items], dtype=np.int32)

    def __len__(self):
        return len(self.agent_id)

    def __getitem__(self, agent_id):
        return self._items[int(agent_id)]


class BehaviorSpec:
    def __init__(self, obs_dim, branches):
        self.observation_specs = [("vector", (obs_dim,))]
        self.action_spec = f"discrete branches={tuple(branches)}"
