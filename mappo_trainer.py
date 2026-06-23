"""
MAPPO Trainer for "The Thing" Social Deduction Game
====================================================
Unity 빌드와 mlagents-envs로 통신하며 MAPPO 학습을 수행하는 트레이너.

구조:
  - Actor: 역할별 3개 (Human, Saboteur, Captain)
  - Centralized Critic: 팀별 2개 (Human팀, Saboteur팀)
  - RolloutBuffer: 에이전트별 경험 저장
  - MAPPOTrainer: 학습 루프 관리

사용법:
  conda activate marl
  python mappo_trainer.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple


# ================================================================
# 설정
# ================================================================

CONFIG = {
    # === 환경 ===
    "env_path": "Builds/Linux/THe_thing.x86_64",
    "no_graphics": True,          # 헤드리스 모드 (화면 없이)
    "time_scale": 20.0,           # 게임 속도 배율 (높을수록 빠름)

    # === 관측/행동 공간 ===
    "obs_dim": 42,                # CollectObservations에서 정의한 크기
    "action_branches": [9, 4],    # [방 선택(0-8), 상호작용(0-3)]

    # === 팀 구성 ===
    "max_human_team": 4,          # 인간3 + 함장1
    "max_saboteur_team": 2,       # 사보타주2

    # === 네트워크 ===
    "hidden_dim": 128,            # 은닉층 크기
    "num_layers": 2,              # 은닉층 수

    # === PPO 하이퍼파라미터 ===
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "gamma": 0.99,                # 할인율
    "gae_lambda": 0.95,           # GAE lambda
    "clip_epsilon": 0.2,          # PPO 클리핑
    "entropy_coef": 0.01,         # 엔트로피 보너스 (탐색 장려)
    "value_coef": 0.5,            # 가치 손실 가중치
    "max_grad_norm": 0.5,         # 그래디언트 클리핑

    # === 학습 ===
    "total_timesteps": 1_000_000,
    "rollout_length": 512,        # 한 번에 수집할 스텝 수
    "ppo_epochs": 4,              # 수집한 데이터로 몇 번 업데이트
    "mini_batch_size": 128,       # 미니배치 크기

    # === 저장/로깅 ===
    "save_dir": "checkpoints",
    "log_dir": "runs/mappo",
    "save_interval": 50,          # N 에피소드마다 저장
    "log_interval": 10,           # N 에피소드마다 로그
}


# ================================================================
# Actor 네트워크 (역할별 정책)
# ================================================================

class Actor(nn.Module):
    """
    관측(42차원)을 받아서 행동 확률을 출력.
    행동 브랜치가 2개(방 선택, 상호작용)이므로 헤드도 2개.
    """

    def __init__(self, obs_dim, action_branches, hidden_dim, num_layers=2):
        super().__init__()

        # 공유 레이어
        layers = []
        input_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        # 브랜치별 출력 헤드
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, branch_size)
            for branch_size in action_branches
        ])

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # 출력 헤드는 작은 값으로
        for head in self.heads:
            nn.init.orthogonal_(head.weight, gain=0.01)

    def forward(self, obs):
        """logits 리스트 반환 (브랜치별)"""
        x = self.shared(obs)
        return [head(x) for head in self.heads]

    def get_action(self, obs):
        """행동 샘플링 + log_prob 반환"""
        logits_list = self.forward(obs)

        actions = []
        log_probs = []

        for logits in logits_list:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            actions.append(action)
            log_probs.append(dist.log_prob(action))

        # actions: (batch, 2), log_probs: (batch,) - 브랜치 합산
        return (
            torch.stack(actions, dim=-1),
            torch.stack(log_probs, dim=-1).sum(dim=-1),
        )

    def evaluate(self, obs, actions):
        """저장된 행동에 대한 log_prob, entropy 계산 (PPO 업데이트용)"""
        logits_list = self.forward(obs)

        log_probs = []
        entropies = []

        for i, logits in enumerate(logits_list):
            dist = torch.distributions.Categorical(logits=logits)
            log_probs.append(dist.log_prob(actions[:, i]))
            entropies.append(dist.entropy())

        return (
            torch.stack(log_probs, dim=-1).sum(dim=-1),
            torch.stack(entropies, dim=-1).sum(dim=-1),
        )


# ================================================================
# Centralized Critic (팀별 가치 함수)
# ================================================================

class CentralizedCritic(nn.Module):
    """
    MAPPO의 핵심: 팀원 전체의 관측을 concat해서 가치 추정.
    - Human팀 critic: 인간3 + 함장1 = 최대 4명의 관측 (4 * 42 = 168)
    - Saboteur팀 critic: 사보타주2 = 최대 2명의 관측 (2 * 42 = 84)

    에이전트가 죽으면 해당 슬롯은 0으로 패딩.
    """

    def __init__(self, obs_dim, max_team_size, hidden_dim, num_layers=2):
        super().__init__()

        self.obs_dim = obs_dim
        self.max_team_size = max_team_size
        input_dim = obs_dim * max_team_size

        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, team_obs):
        """
        team_obs: (batch, max_team_size * obs_dim)
        반환: (batch,) 가치 추정
        """
        return self.net(team_obs).squeeze(-1)


# ================================================================
# Rollout Buffer (경험 저장소)
# ================================================================

class RolloutBuffer:
    """에이전트 한 명의 rollout 데이터를 저장"""

    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.team_obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, team_obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.team_obs.append(team_obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def __len__(self):
        return len(self.obs)

    def compute_gae(self, last_value, gamma, gae_lambda):
        """
        Generalized Advantage Estimation 계산.
        반환: (returns, advantages) 리스트
        """
        rewards = self.rewards
        dones = self.dones
        values = self.values + [last_value]

        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return returns, advantages

    def to_tensors(self, returns, advantages, device):
        """numpy/list → PyTorch 텐서 변환"""
        return {
            "obs": torch.FloatTensor(np.array(self.obs)).to(device),
            "team_obs": torch.FloatTensor(np.array(self.team_obs)).to(device),
            "actions": torch.LongTensor(np.array(self.actions)).to(device),
            "log_probs": torch.FloatTensor(np.array(self.log_probs)).to(device),
            "returns": torch.FloatTensor(np.array(returns)).to(device),
            "advantages": torch.FloatTensor(np.array(advantages)).to(device),
        }


# ================================================================
# MAPPO Trainer (메인 학습 루프)
# ================================================================

class MAPPOTrainer:
    """
    전체 학습을 관리하는 메인 클래스.

    워크플로우:
    1. Unity 환경 실행
    2. 에이전트별 관측 수신 → 역할 판별 → 해당 Actor로 행동 선택
    3. 행동을 Unity로 전송 → 보상 수신
    4. rollout_length만큼 모이면 PPO 업데이트
    5. 반복
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[장치] {self.device}")

        # --- 역할별 Actor ---
        self.actors = {
            "human": Actor(
                config["obs_dim"], config["action_branches"],
                config["hidden_dim"], config["num_layers"]
            ).to(self.device),
            "saboteur": Actor(
                config["obs_dim"], config["action_branches"],
                config["hidden_dim"], config["num_layers"]
            ).to(self.device),
            "captain": Actor(
                config["obs_dim"], config["action_branches"],
                config["hidden_dim"], config["num_layers"]
            ).to(self.device),
        }

        # --- 팀별 Centralized Critic ---
        self.critics = {
            "human_team": CentralizedCritic(
                config["obs_dim"], config["max_human_team"],
                config["hidden_dim"], config["num_layers"]
            ).to(self.device),
            "saboteur_team": CentralizedCritic(
                config["obs_dim"], config["max_saboteur_team"],
                config["hidden_dim"], config["num_layers"]
            ).to(self.device),
        }

        # --- Optimizer ---
        self.actor_optimizers = {
            role: optim.Adam(actor.parameters(), lr=config["lr_actor"])
            for role, actor in self.actors.items()
        }
        self.critic_optimizers = {
            team: optim.Adam(critic.parameters(), lr=config["lr_critic"])
            for team, critic in self.critics.items()
        }

        # --- 에이전트별 버퍼 ---
        self.buffers = defaultdict(RolloutBuffer)

        # --- 에이전트 역할 매핑 (에피소드 시작 시 설정) ---
        self.agent_roles = {}     # agent_id → 'human'/'saboteur'/'captain'
        self.agent_teams = {}     # agent_id → 'human_team'/'saboteur_team'

        # --- 통계 ---
        self.episode_rewards = defaultdict(float)
        self.episode_count = 0
        self.total_steps = 0

        # --- TensorBoard ---
        self.writer = SummaryWriter(config["log_dir"])

        # --- 체크포인트 ---
        os.makedirs(config["save_dir"], exist_ok=True)

    # ============================================================
    # 역할 판별
    # ============================================================

    def identify_role(self, obs):
        """
        관측 벡터에서 역할 판별.
        obs[0] = isSaboteur (1 or 0)
        obs[1] = isCaptain (1 or 0)
        """
        if obs[1] > 0.5:
            return "captain"
        elif obs[0] > 0.5:
            return "saboteur"
        else:
            return "human"

    def get_team(self, role):
        """역할 → 팀"""
        return "saboteur_team" if role == "saboteur" else "human_team"

    # ============================================================
    # 팀 관측 구성 (Centralized Critic용)
    # ============================================================

    def build_team_obs(self, team, all_agent_obs):
        """
        팀원 전체의 관측을 concat하여 centralized critic 입력 생성.
        죽거나 없는 슬롯은 0 패딩.

        all_agent_obs: {agent_id: obs_numpy} (현재 살아있는 에이전트)
        team: 'human_team' 또는 'saboteur_team'
        """
        obs_dim = self.config["obs_dim"]

        if team == "human_team":
            max_size = self.config["max_human_team"]
        else:
            max_size = self.config["max_saboteur_team"]

        # 해당 팀 에이전트들의 관측 수집
        team_observations = []
        for agent_id, obs in all_agent_obs.items():
            if agent_id in self.agent_teams and self.agent_teams[agent_id] == team:
                team_observations.append(obs)

        # 패딩
        result = np.zeros(max_size * obs_dim, dtype=np.float32)
        for i, obs in enumerate(team_observations):
            if i < max_size:
                start = i * obs_dim
                end = start + obs_dim
                result[start:end] = obs

        return result

    # ============================================================
    # 행동 선택
    # ============================================================

    @torch.no_grad()
    def select_action(self, agent_id, obs_numpy, team_obs_numpy):
        """
        에이전트의 관측으로 행동 선택.
        반환: (action_numpy, log_prob_float, value_float)
        """
        role = self.agent_roles[agent_id]
        team = self.agent_teams[agent_id]

        obs_tensor = torch.FloatTensor(obs_numpy).unsqueeze(0).to(self.device)
        team_obs_tensor = torch.FloatTensor(team_obs_numpy).unsqueeze(0).to(self.device)

        # Actor: 행동 선택
        action, log_prob = self.actors[role].get_action(obs_tensor)

        # Critic: 가치 추정
        value = self.critics[team](team_obs_tensor)

        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    # ============================================================
    # PPO 업데이트
    # ============================================================

    def update(self):
        """
        수집된 rollout으로 Actor/Critic 업데이트.
        역할별, 팀별로 분리해서 업데이트.
        """
        config = self.config

        # --- 역할별 데이터 수집 ---
        role_data = defaultdict(list)   # role → [tensor_dict, ...]
        team_data = defaultdict(list)   # team → [tensor_dict, ...]

        for agent_id, buffer in self.buffers.items():
            if len(buffer) == 0:
                continue

            role = self.agent_roles.get(agent_id)
            team = self.agent_teams.get(agent_id)
            if role is None or team is None:
                continue

            # 마지막 value 추정 (bootstrap)
            last_value = 0.0  # 에피소드 끝나면 0
            if not buffer.dones[-1]:
                # 아직 안 끝남 → 현재 value로 bootstrap
                last_value = buffer.values[-1]

            returns, advantages = buffer.compute_gae(
                last_value, config["gamma"], config["gae_lambda"]
            )
            tensors = buffer.to_tensors(returns, advantages, self.device)
            tensors["role"] = role
            tensors["team"] = team

            role_data[role].append(tensors)
            team_data[team].append(tensors)

        # --- Actor 업데이트 (역할별) ---
        for role, data_list in role_data.items():
            if len(data_list) == 0:
                continue

            # 모든 에이전트 데이터 합치기
            all_obs = torch.cat([d["obs"] for d in data_list])
            all_actions = torch.cat([d["actions"] for d in data_list])
            all_old_log_probs = torch.cat([d["log_probs"] for d in data_list])
            all_advantages = torch.cat([d["advantages"] for d in data_list])

            # Advantage 정규화
            if len(all_advantages) > 1:
                all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

            actor = self.actors[role]
            optimizer = self.actor_optimizers[role]

            for _ in range(config["ppo_epochs"]):
                # 미니배치 생성
                indices = np.arange(len(all_obs))
                np.random.shuffle(indices)

                for start in range(0, len(indices), config["mini_batch_size"]):
                    end = start + config["mini_batch_size"]
                    mb_idx = indices[start:end]

                    mb_obs = all_obs[mb_idx]
                    mb_actions = all_actions[mb_idx]
                    mb_old_log_probs = all_old_log_probs[mb_idx]
                    mb_advantages = all_advantages[mb_idx]

                    # 새 log_prob, entropy
                    new_log_probs, entropy = actor.evaluate(mb_obs, mb_actions)

                    # PPO 클리핑
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - config["clip_epsilon"], 1 + config["clip_epsilon"]) * mb_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # 엔트로피 보너스
                    entropy_loss = -entropy.mean()

                    # 총 손실
                    loss = actor_loss + config["entropy_coef"] * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), config["max_grad_norm"])
                    optimizer.step()

        # --- Critic 업데이트 (팀별) ---
        for team, data_list in team_data.items():
            if len(data_list) == 0:
                continue

            all_team_obs = torch.cat([d["team_obs"] for d in data_list])
            all_returns = torch.cat([d["returns"] for d in data_list])

            critic = self.critics[team]
            optimizer = self.critic_optimizers[team]

            for _ in range(config["ppo_epochs"]):
                indices = np.arange(len(all_team_obs))
                np.random.shuffle(indices)

                for start in range(0, len(indices), config["mini_batch_size"]):
                    end = start + config["mini_batch_size"]
                    mb_idx = indices[start:end]

                    mb_team_obs = all_team_obs[mb_idx]
                    mb_returns = all_returns[mb_idx]

                    values = critic(mb_team_obs)
                    critic_loss = config["value_coef"] * ((values - mb_returns) ** 2).mean()

                    optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), config["max_grad_norm"])
                    optimizer.step()

        # --- 버퍼 초기화 ---
        for buffer in self.buffers.values():
            buffer.clear()

    # ============================================================
    # 저장/불러오기
    # ============================================================

    def save(self, path=None):
        if path is None:
            path = os.path.join(
                self.config["save_dir"],
                f"mappo_ep{self.episode_count}.pt"
            )

        checkpoint = {
            "episode": self.episode_count,
            "total_steps": self.total_steps,
        }

        for role, actor in self.actors.items():
            checkpoint[f"actor_{role}"] = actor.state_dict()
            checkpoint[f"actor_opt_{role}"] = self.actor_optimizers[role].state_dict()

        for team, critic in self.critics.items():
            checkpoint[f"critic_{team}"] = critic.state_dict()
            checkpoint[f"critic_opt_{team}"] = self.critic_optimizers[team].state_dict()

        torch.save(checkpoint, path)
        print(f"[저장] {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.episode_count = checkpoint.get("episode", 0)
        self.total_steps = checkpoint.get("total_steps", 0)

        for role, actor in self.actors.items():
            if f"actor_{role}" in checkpoint:
                actor.load_state_dict(checkpoint[f"actor_{role}"])
                self.actor_optimizers[role].load_state_dict(checkpoint[f"actor_opt_{role}"])

        for team, critic in self.critics.items():
            if f"critic_{team}" in checkpoint:
                critic.load_state_dict(checkpoint[f"critic_{team}"])
                self.critic_optimizers[team].load_state_dict(checkpoint[f"critic_opt_{team}"])

        print(f"[불러오기] {path} (에피소드: {self.episode_count})")

    # ============================================================
    # 메인 학습 루프
    # ============================================================

    def train(self):
        """메인 학습 루프"""
        config = self.config

        # --- Unity 환경 시작 ---
        print("=" * 50)
        print("[시작] Unity 환경 로딩 중...")
        print(f"  빌드 경로: {config['env_path']}")
        print(f"  헤드리스: {config['no_graphics']}")
        print(f"  시간 배율: {config['time_scale']}x")
        print("=" * 50)

        channel = EngineConfigurationChannel()
        env = UnityEnvironment(
            file_name=config["env_path"],
            side_channels=[channel],
            no_graphics=config["no_graphics"],
        )
        channel.set_configuration_parameters(time_scale=config["time_scale"])

        env.reset()

        # Behavior 이름 확인
        behavior_names = list(env.behavior_specs.keys())
        print(f"[환경] Behavior names: {behavior_names}")

        if len(behavior_names) == 0:
            print("[오류] Behavior가 없습니다. Unity Agent 설정을 확인하세요.")
            env.close()
            return

        behavior_name = behavior_names[0]
        spec = env.behavior_specs[behavior_name]
        print(f"[환경] 관측 shape: {spec.observation_specs}")
        print(f"[환경] 행동 spec: {spec.action_spec}")

        # --- 학습 루프 ---
        steps_since_update = 0
        episode_start_time = time.time()
        terminated_agents = set()

        try:
            while self.total_steps < config["total_timesteps"]:

                # 현재 스텝의 에이전트 상태 가져오기
                decision_steps, terminal_steps = env.get_steps(behavior_name)

                # ---- 종료된 에이전트 처리 ----
                for agent_id in terminal_steps.agent_id:
                    agent_id = int(agent_id)

                    # 마지막 보상 기록
                    reward = terminal_steps[agent_id].reward
                    self.episode_rewards[agent_id] += reward

                    # 버퍼에 종료 표시
                    if agent_id in self.agent_roles and len(self.buffers[agent_id]) > 0:
                        self.buffers[agent_id].rewards[-1] += reward
                        self.buffers[agent_id].dones[-1] = 1.0

                    terminated_agents.add(agent_id)

                # ---- 에피소드 종료 감지 ----
                # 등록된 에이전트 전원이 terminated이거나, decision이 0인데 terminal이 있으면 종료
                all_known_done = (
                    len(self.agent_roles) > 0
                    and set(self.agent_roles.keys()).issubset(terminated_agents)
                )
                if all_known_done or (len(decision_steps) == 0 and len(terminal_steps) > 0):
                    self.episode_count += 1

                    # 남은 버퍼로 마지막 업데이트
                    has_data = any(len(buf) > 0 for buf in self.buffers.values())
                    if has_data:
                        self.update()

                    # 통계 기록
                    total_ep_reward = sum(self.episode_rewards.values())
                    elapsed = time.time() - episode_start_time

                    if self.episode_count % config["log_interval"] == 0:
                        print(
                            f"[에피소드 {self.episode_count}] "
                            f"총 보상: {total_ep_reward:.2f} | "
                            f"스텝: {self.total_steps} | "
                            f"시간: {elapsed:.1f}s"
                        )

                    # TensorBoard 기록
                    self.writer.add_scalar("episode/total_reward", total_ep_reward, self.episode_count)
                    self.writer.add_scalar("episode/length", steps_since_update, self.episode_count)

                    # 역할별 보상 기록
                    for agent_id, reward in self.episode_rewards.items():
                        role = self.agent_roles.get(agent_id, "unknown")
                        self.writer.add_scalar(f"reward/{role}", reward, self.episode_count)

                    # 저장
                    if self.episode_count % config["save_interval"] == 0:
                        self.save()

                    # 초기화
                    self.episode_rewards.clear()
                    self.buffers.clear()
                    terminated_agents.clear()
                    episode_start_time = time.time()
                    steps_since_update = 0

                    # 환경 리셋
                    env.reset()
                    self.agent_roles.clear()
                    self.agent_teams.clear()
                    continue

                # ---- 결정이 필요한 에이전트 처리 ----
                if len(decision_steps) == 0:
                    env.step()
                    continue

                # 현재 살아있는 에이전트들의 관측 수집
                all_agent_obs = {}
                for agent_id in decision_steps.agent_id:
                    agent_id = int(agent_id)
                    obs = decision_steps[agent_id].obs[0]  # 첫 번째 관측
                    all_agent_obs[agent_id] = obs

                    # 역할 판별 (첫 스텝에서)
                    if agent_id not in self.agent_roles:
                        role = self.identify_role(obs)
                        self.agent_roles[agent_id] = role
                        self.agent_teams[agent_id] = self.get_team(role)

                # 팀 관측 구성 (centralized critic용)
                team_obs_cache = {
                    "human_team": self.build_team_obs("human_team", all_agent_obs),
                    "saboteur_team": self.build_team_obs("saboteur_team", all_agent_obs),
                }

                # 각 에이전트별 행동 선택
                actions_dict = {}
                for agent_id in decision_steps.agent_id:
                    agent_id = int(agent_id)
                    obs = all_agent_obs[agent_id]
                    team = self.agent_teams[agent_id]
                    team_obs = team_obs_cache[team]

                    action, log_prob, value = self.select_action(agent_id, obs, team_obs)

                    # 행동 저장
                    actions_dict[agent_id] = action

                    # 보상 (이전 스텝의)
                    reward = decision_steps[agent_id].reward
                    self.episode_rewards[agent_id] += reward

                    # 버퍼에 추가
                    self.buffers[agent_id].add(
                        obs=obs,
                        team_obs=team_obs,
                        action=action,
                        log_prob=log_prob,
                        reward=reward,
                        done=0.0,
                        value=value,
                    )

                # 행동을 Unity에 전송
                action_array = np.zeros(
                    (len(decision_steps), len(config["action_branches"])),
                    dtype=np.int32
                )
                for i, agent_id in enumerate(decision_steps.agent_id):
                    agent_id = int(agent_id)
                    if agent_id in actions_dict:
                        action_array[i] = actions_dict[agent_id]

                action_tuple = ActionTuple(discrete=action_array)
                env.set_actions(behavior_name, action_tuple)
                env.step()

                self.total_steps += len(decision_steps)
                steps_since_update += len(decision_steps)

                # ---- 업데이트 체크 ----
                # 살아있는 에이전트 기준으로만 판단 (사망 에이전트의 짧은 버퍼가 블로킹하지 않도록)
                living_agents = set(self.agent_roles.keys()) - terminated_agents
                if living_agents:
                    living_buf_lens = [
                        len(self.buffers[aid]) for aid in living_agents
                        if len(self.buffers[aid]) > 0
                    ]
                    if living_buf_lens and min(living_buf_lens) >= config["rollout_length"]:
                        self.update()
                        steps_since_update = 0

        except KeyboardInterrupt:
            print("\n[중단] 학습이 중단되었습니다.")
            self.save(os.path.join(config["save_dir"], "mappo_interrupted.pt"))

        finally:
            env.close()
            self.writer.close()
            print(f"[완료] 총 {self.total_steps} 스텝, {self.episode_count} 에피소드")


# ================================================================
# 실행
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MAPPO Trainer for The Thing")
    parser.add_argument("--load", type=str, default=None, help="체크포인트 경로")
    parser.add_argument("--timesteps", type=int, default=None, help="총 학습 스텝")
    parser.add_argument("--env-path", type=str, default=None, help="Unity 빌드 경로")
    parser.add_argument("--time-scale", type=float, default=None, help="게임 속도 배율")
    parser.add_argument("--graphics", action="store_true", help="화면 표시 (디버그용)")
    args = parser.parse_args()

    # 커맨드라인 인자로 설정 덮어쓰기
    if args.timesteps:
        CONFIG["total_timesteps"] = args.timesteps
    if args.env_path:
        CONFIG["env_path"] = args.env_path
    if args.time_scale:
        CONFIG["time_scale"] = args.time_scale
    if args.graphics:
        CONFIG["no_graphics"] = False

    # 트레이너 생성 및 실행
    trainer = MAPPOTrainer(CONFIG)

    if args.load:
        trainer.load(args.load)

    trainer.train()
