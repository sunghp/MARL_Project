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
import yaml
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from mlagents_envs.base_env import ActionTuple


# ================================================================
# 설정
# ================================================================

CONFIG = {
    # === 환경 ===
    "env_path": "Builds/Windows/My project.exe",
    "balance_yaml": "config/TheThing.yaml",   # 게임 밸런스 파라미터 yaml (environment_parameters 블록을 읽어 Unity에 주입)
    "no_graphics": True,          # 헤드리스 모드 (화면 없이)
    "time_scale": 20.0,           # 게임 속도 배율 (높을수록 빠름)

    # === 관측/행동 공간 ===
    "local_obs_dim": 64,     # Actor 입력 (부분 관측)
    "global_obs_dim": 56,    # Critic 입력 (전역, per-agent)
    "action_branches": [9, 9],    # [방 선택(0-8), 상호작용(0 없음,1 부수기,2 고치기,3-8 슬롯 사격)]

    # === 팀 구성 ===
    "max_human_team": 4,          # 인간3 + 함장1
    "max_saboteur_team": 2,       # 사보타주2

    # === 네트워크 ===
    "hidden_dim": 128,            # 은닉층 크기
    "num_layers": 2,              # 은닉층 수

    # === PPO 하이퍼파라미터 ===
    "lr_actor": 1e-4,
    "lr_critic": 1e-4,
    "gamma": 0.995,               # 할인율 (긴 에피소드 + 끝보상 환경이라 0.99보다 길게)
    "gae_lambda": 0.95,           # GAE lambda
    "clip_epsilon": 0.2,          # PPO 클리핑
    "entropy_coef": 0.01,         # 엔트로피 보너스 (탐색 장려)
    "value_coef": 0.5,            # 가치 손실 가중치
    "max_grad_norm": 0.5,         # 그래디언트 클리핑

    # === Unity 보상 상수 (NPCAgent.winReward / loseReward와 동일하게) ===
    # 죽은 에이전트는 게임 종료 보상을 직접 못 받으므로 트레이너가 마지막 전이에 대신 붙인다
    "win_reward": 5.0,
    "lose_reward": -5.0,

    # === 학습 ===
    "total_timesteps": 2_000_000,
    "rollout_length": 512,        # 한 번에 수집할 스텝 수
    "ppo_epochs": 4,              # 수집한 데이터로 몇 번 업데이트
    "mini_batch_size": 128,       # 미니배치 크기

    # === 저장/로깅 ===
    "save_dir": "checkpoints",
    "log_dir": "runs/mappo",
    "save_interval": 50,          # N 에피소드마다 저장
    "log_interval": 10,           # N 에피소드마다 로그

    # === 고정상대 평가 ===
    "eval_interval": 100000,     # N 스텝마다 평가 (10만)
    "eval_episodes": 30,          # 각 매치업(사보타주/인간)당 평가 에피소드 수
}


# ================================================================
# Actor 네트워크 (역할별 정책)
# ================================================================

class Actor(nn.Module):
    """
    로컬 관측(64차원)을 받아서 행동 확률을 출력.
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
    - Human팀 critic: 인간3 + 함장1 = 최대 4명의 관측 (4 * 56 = 224)
    - Saboteur팀 critic: 사보타주2 = 최대 2명의 관측 (2 * 56 = 112)

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

    def compute_gae(self, last_value, gamma, gae_lambda, n=None):
        """
        Generalized Advantage Estimation 계산 (앞 n개 전이만).
        last_value: n번째 전이 다음 상태의 가치 V(s_n) (종료면 0)
        반환: (returns, advantages) 리스트
        """
        if n is None:
            n = len(self.rewards)
        rewards = self.rewards[:n]
        dones = self.dones[:n]
        values = self.values[:n] + [last_value]

        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return returns, advantages

    def drop_first(self, n):
        """앞 n개 전이 삭제 (학습에 쓴 부분만 버리고 나머지는 다음 rollout으로 이월)"""
        for lst in (self.obs, self.team_obs, self.actions, self.log_probs,
                    self.rewards, self.dones, self.values):
            del lst[:n]

    def to_tensors(self, returns, advantages, device, n=None):
        """numpy/list → PyTorch 텐서 변환 (앞 n개 전이만)"""
        if n is None:
            n = len(self.obs)
        return {
            "obs": torch.FloatTensor(np.array(self.obs[:n])).to(device),
            "team_obs": torch.FloatTensor(np.array(self.team_obs[:n])).to(device),
            "actions": torch.LongTensor(np.array(self.actions[:n])).to(device),
            "log_probs": torch.FloatTensor(np.array(self.log_probs[:n])).to(device),
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
                config["local_obs_dim"], config["action_branches"],
                config["hidden_dim"], config["num_layers"]
            ).to(self.device),
            "saboteur": Actor(
                config["local_obs_dim"], config["action_branches"],
                config["hidden_dim"], config["num_layers"]
            ).to(self.device),
            "captain": Actor(
                config["local_obs_dim"], config["action_branches"],
                config["hidden_dim"], config["num_layers"]
            ).to(self.device),
        }

        # --- 팀별 Centralized Critic ---
        self.critics = {
            "human_team": CentralizedCritic(
                config["global_obs_dim"], config["max_human_team"],
                config["hidden_dim"], config["num_layers"]
            ).to(self.device),
            "saboteur_team": CentralizedCritic(
                config["global_obs_dim"], config["max_saboteur_team"],
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
        self.total_steps = 0      # 학습 스텝 (평가 중 스텝은 제외)
        self.eval_steps = 0       # 평가에 쓴 스텝

        # --- 고정상대 평가 상태 ---
        self.in_eval = False        # 평가 중 여부
        self.eval_frozen = 0        # 현재 고정 팀 (2=인간고정→사보타주측정, 1=사보타주고정→인간측정)
        self.eval_ep_count = 0      # 현재 매치업에서 집계된 평가 에피소드 수
        self.next_eval_step = self.config["eval_interval"]  # 다음 평가 트리거 스텝

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
        obs_dim = self.config["global_obs_dim"]

        if team == "human_team":
            max_size = self.config["max_human_team"]
        else:
            max_size = self.config["max_saboteur_team"]

        # 해당 팀 에이전트들의 관측 수집
        # [수정] agent_id 순으로 정렬 → 팀원이 죽어도 슬롯 순서가 흔들리지 않음
        team_observations = []
        for agent_id in sorted(all_agent_obs.keys()):
            if agent_id in self.agent_teams and self.agent_teams[agent_id] == team:
                team_observations.append(all_agent_obs[agent_id])

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
        used_counts = {}                # agent_id → 학습에 사용한 전이 수

        for agent_id, buffer in self.buffers.items():
            if len(buffer) == 0:
                continue

            role = self.agent_roles.get(agent_id)
            team = self.agent_teams.get(agent_id)
            if role is None or team is None:
                continue

            if buffer.dones[-1]:
                # 에피소드 종료 → 전체 사용, bootstrap 0
                n_train, last_value = len(buffer), 0.0
            else:
                # 에피소드 진행 중: 마지막 전이의 보상은 다음 decision step에 도착하므로 아직 미완성.
                # 마지막 전이는 다음 rollout으로 이월하고, 그 상태의 V(s)로 직전까지를 bootstrap.
                n_train, last_value = len(buffer) - 1, buffer.values[-1]
            if n_train == 0:
                continue
            used_counts[agent_id] = n_train

            returns, advantages = buffer.compute_gae(
                last_value, config["gamma"], config["gae_lambda"], n_train
            )
            tensors = buffer.to_tensors(returns, advantages, self.device, n_train)
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
            loss_acc, ent_acc, n_batches = 0.0, 0.0, 0

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

                    loss_acc += actor_loss.item()
                    ent_acc += entropy.mean().item()
                    n_batches += 1

            if n_batches > 0:
                self.writer.add_scalar(f"loss/actor_{role}", loss_acc / n_batches, self.total_steps)
                self.writer.add_scalar(f"policy/entropy_{role}", ent_acc / n_batches, self.total_steps)

        # --- Critic 업데이트 (팀별) ---
        for team, data_list in team_data.items():
            if len(data_list) == 0:
                continue

            all_team_obs = torch.cat([d["team_obs"] for d in data_list])
            all_returns = torch.cat([d["returns"] for d in data_list])

            critic = self.critics[team]
            optimizer = self.critic_optimizers[team]
            closs_acc, cn = 0.0, 0

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

                    closs_acc += critic_loss.item()
                    cn += 1

            if cn > 0:
                self.writer.add_scalar(f"loss/critic_{team}", closs_acc / cn, self.total_steps)

        # --- 버퍼 정리: 학습에 쓴 전이만 삭제 (미완성 마지막 전이는 이월) ---
        for agent_id, buffer in self.buffers.items():
            if agent_id in used_counts:
                buffer.drop_first(used_counts[agent_id])
            elif len(buffer) > 0 and buffer.dones[-1]:
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
        param_channel = EnvironmentParametersChannel()
        stats_channel = StatsSideChannel()   # Unity 행동 지표 수신용

        # --- 게임 밸런스 파라미터 로드 (yaml의 environment_parameters 블록) ---
        env_params = {}
        balance_path = config.get("balance_yaml")
        if balance_path and os.path.exists(balance_path):
            with open(balance_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            env_params = loaded.get("environment_parameters", {}) or {}
            print(f"[밸런스] {balance_path}에서 {len(env_params)}개 파라미터 로드")
        else:
            print(f"[밸런스] yaml을 찾지 못함({balance_path}) → Unity Inspector 기본값 사용")

        env = UnityEnvironment(
            file_name=config["env_path"],
            side_channels=[channel, param_channel, stats_channel],
            base_port=5004,
            timeout_wait=120,
            no_graphics=config["no_graphics"],
        )
        channel.set_configuration_parameters(time_scale=config["time_scale"])

        # --- 밸런스 파라미터 주입 → Unity GameManager.LoadEnvironmentParameters()가 받음 ---
        for key, val in env_params.items():
            param_channel.set_float_parameter(key, float(val))
            print(f"  - {key} = {float(val)}")

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
        episode_steps = 0          # [수정] 실제 에피소드 길이 (업데이트로 리셋되지 않음)
        episode_start_time = time.time()
        terminated_agents = set()
        died_early = set()         # 게임 종료 전에 죽어서 terminal을 먼저 받은 에이전트

        try:
            while self.total_steps < config["total_timesteps"]:

                # ---- 평가 트리거: N 스텝마다 학습 멈추고 고정상대 평가 진입 ----
                if not self.in_eval and self.total_steps >= self.next_eval_step:
                    self.in_eval = True
                    self.eval_frozen = 2          # 인간 고정 → 학습된 사보타주 측정부터
                    self.eval_ep_count = 0
                    param_channel.set_float_parameter("eval_mode", 1.0)
                    param_channel.set_float_parameter("frozen_team", 2.0)
                    print(f"[평가 시작] step={self.total_steps} | 사보타주 측정 (인간 규칙봇 고정)")

                # 현재 스텝의 에이전트 상태 가져오기
                decision_steps, terminal_steps = env.get_steps(behavior_name)

                # ---- 종료된 에이전트 처리 ----
                # ML-Agents 동작: 죽어서 SetActive(false)된 에이전트는 그 즉시 terminal을 보내고,
                # 게임 종료 시 재활성화되면 "새 episode id"를 받는다. 그래서 게임 종료 exchange에는
                # 이번 에피소드에 decision이 한 번도 없던 id(= 시체)의 terminal이 섞여 온다.
                # 이 terminal은 리셋 후 관측/역할이라 학습에 쓸 수 없으므로 개수만 센다.
                corpse_terminals = 0
                term_progress = {}
                for agent_id in terminal_steps.agent_id:
                    agent_id = int(agent_id)

                    if agent_id not in self.agent_roles:
                        corpse_terminals += 1
                        continue
                    # obs[6] = 항해 진행도: 사망 terminal은 리셋 전(>0), 게임 종료 terminal은 리셋 후(0)
                    term_progress[agent_id] = float(terminal_steps[agent_id].obs[0][6])

                    # 마지막 보상 기록
                    reward = terminal_steps[agent_id].reward
                    self.episode_rewards[agent_id] += reward

                    # 버퍼에 종료 표시
                    if agent_id in self.agent_roles and len(self.buffers[agent_id]) > 0:
                        self.buffers[agent_id].rewards[-1] += reward
                        self.buffers[agent_id].dones[-1] = 1.0

                    terminated_agents.add(agent_id)

                # ---- 에피소드 종료 감지 ----
                # 등록된 에이전트 전원이 terminated일 때만 종료.
                # (decision 없이 terminal만 온 exchange는 "누가 죽음"일 뿐 게임 종료가 아니다.
                #  예전엔 이걸 종료로 보고 env.reset()을 불러 게임 도중에 에피소드를 잘랐다.)
                all_known_done = (
                    len(self.agent_roles) > 0
                    and set(self.agent_roles.keys()).issubset(terminated_agents)
                )
                if not all_known_done:
                    died_early.update(term_progress.keys())
                if all_known_done:
                    self.episode_count += 1

                    elapsed = time.time() - episode_start_time
                    env_stats = stats_channel.get_and_reset_stats()

                    # 승패 결과 (Unity GameManager가 outcome/human_win으로 보고)
                    outcome = env_stats.get("outcome/human_win")
                    human_won = (outcome[-1][0] > 0.5) if outcome else None

                    # 죽은 에이전트에게 게임 종료 보상 귀속 (재활성화된 새 id 쪽으로 가버린 보상 복구)
                    if human_won is not None:
                        dead = set(died_early)
                        # 같은 exchange에서 죽고 곧바로 게임이 끝난 경우(결정타 사격):
                        # 시체 terminal 수 - 이전 사망자 수만큼, 리셋 전 관측(진행도 > 0)인 에이전트를 사망자로 판정
                        n_same = corpse_terminals - len(died_early)
                        if n_same > 0:
                            cands = sorted(
                                (aid for aid in term_progress if aid not in dead),
                                key=lambda aid: term_progress[aid], reverse=True,
                            )
                            dead.update(aid for aid in cands[:n_same] if term_progress[aid] > 0)
                        for aid in dead:
                            team_won = human_won if self.agent_teams[aid] == "human_team" else not human_won
                            r = config["win_reward"] if team_won else config["lose_reward"]
                            self.episode_rewards[aid] += r
                            if len(self.buffers[aid]) > 0:
                                self.buffers[aid].rewards[-1] += r

                    # 파라미터 반영이 한 에피소드 늦기 때문에(Unity는 이전 에피소드 종료 시 리셋하며 읽음)
                    # 평가 종료 직후 에피소드는 아직 규칙봇이 섞인 평가 모드로 돌았을 수 있음 → 학습에서 제외
                    ran_as_eval = any(k.startswith("eval/") for k in env_stats)

                    if not self.in_eval and ran_as_eval:
                        print("[건너뜀] 평가 모드로 진행된 에피소드 → 학습 데이터에서 제외")
                    elif not self.in_eval:
                        # ===== 학습 에피소드: 업데이트 + 학습 곡선 기록 =====
                        has_data = any(len(buf) > 0 for buf in self.buffers.values())
                        if has_data:
                            self.update()

                        total_ep_reward = sum(self.episode_rewards.values())

                        if self.episode_count % config["log_interval"] == 0:
                            print(
                                f"[에피소드 {self.episode_count}] "
                                f"총 보상: {total_ep_reward:.2f} | "
                                f"스텝: {self.total_steps} | "
                                f"시간: {elapsed:.1f}s"
                            )

                        self.writer.add_scalar("episode/total_reward", total_ep_reward, self.episode_count)
                        self.writer.add_scalar("episode/length", episode_steps, self.episode_count)
                        self.writer.add_scalar("episode/seconds", elapsed, self.episode_count)

                        role_rewards = defaultdict(list)
                        for agent_id, reward in self.episode_rewards.items():
                            role_rewards[self.agent_roles.get(agent_id, "unknown")].append(reward)
                        for role, rewards in role_rewards.items():
                            self.writer.add_scalar(
                                f"reward/{role}", float(np.mean(rewards)), self.episode_count
                            )

                        if human_won is not None:
                            self.writer.add_scalar(
                                "outcome/saboteur_win_rate", 0.0 if human_won else 1.0, self.episode_count
                            )

                        # 행동 지표 + Unity stats (학습 중엔 eval/* 안 옴)
                        for stat_name, entries in env_stats.items():
                            vals = [e[0] for e in entries]
                            if vals:
                                self.writer.add_scalar(
                                    stat_name, float(np.mean(vals)), self.episode_count
                                )

                        if self.episode_count % config["save_interval"] == 0:
                            self.save()
                    else:
                        # ===== 평가 에피소드: 학습 안 함, 승률만 집계 =====
                        # eval/* 통계만 기록 (학습 곡선 오염 방지)
                        for stat_name, entries in env_stats.items():
                            if not stat_name.startswith("eval/"):
                                continue
                            vals = [e[0] for e in entries]
                            if vals:
                                self.writer.add_scalar(
                                    stat_name, float(np.mean(vals)), self.episode_count
                                )

                        # 이 에피소드가 실제 eval이었는지 = 해당 매치업 통계가 들어왔는지
                        eval_key = ("eval/saboteur_vs_bot_win" if self.eval_frozen == 2
                                    else "eval/human_vs_bot_win")
                        if eval_key in env_stats:
                            self.eval_ep_count += 1

                        # 목표 에피소드 수 도달 → 다음 매치업 또는 평가 종료
                        if self.eval_ep_count >= config["eval_episodes"]:
                            if self.eval_frozen == 2:
                                self.eval_frozen = 1        # 사보타주 고정 → 인간 측정
                                self.eval_ep_count = 0
                                param_channel.set_float_parameter("frozen_team", 1.0)
                                print("[평가] 사보타주 측정 완료 → 인간 측정 (사보타주 규칙봇 고정)")
                            else:
                                self.in_eval = False        # 평가 종료 → 학습 재개
                                self.eval_frozen = 0
                                param_channel.set_float_parameter("eval_mode", 0.0)
                                self.next_eval_step = self.total_steps + config["eval_interval"]
                                print(f"[평가 완료] step={self.total_steps} 학습 재개")

                    self.writer.flush()

                    # 초기화
                    self.episode_rewards.clear()
                    self.buffers.clear()
                    terminated_agents.clear()
                    died_early.clear()
                    episode_start_time = time.time()
                    steps_since_update = 0
                    episode_steps = 0

                    # 환경 리셋
                    env.reset()
                    self.agent_roles.clear()
                    self.agent_teams.clear()
                    continue

                # ---- 결정이 필요한 에이전트 처리 ----
                if len(decision_steps) == 0:
                    env.step()
                    continue

                L = config["local_obs_dim"]
                all_agent_local = {}
                all_agent_global = {}
                for agent_id in decision_steps.agent_id:
                    agent_id = int(agent_id)
                    full = decision_steps[agent_id].obs[0]  # 120차원
                    all_agent_local[agent_id] = full[:L]  # 앞 64 = Actor용
                    all_agent_global[agent_id] = full[L:]  # 뒤 56 = Critic용

                    if agent_id not in self.agent_roles:
                        role = self.identify_role(full)  # 앞부분이 self라 그대로 OK
                        self.agent_roles[agent_id] = role
                        self.agent_teams[agent_id] = self.get_team(role)

                # team_obs는 전역 슬라이스로 구성
                team_obs_cache = {
                    "human_team": self.build_team_obs("human_team", all_agent_global),
                    "saboteur_team": self.build_team_obs("saboteur_team", all_agent_global),
                }

                # 각 에이전트별 행동 선택
                actions_dict = {}
                for agent_id in decision_steps.agent_id:
                    agent_id = int(agent_id)
                    local_obs = all_agent_local[agent_id]
                    team = self.agent_teams[agent_id]
                    team_obs = team_obs_cache[team]

                    action, log_prob, value = self.select_action(agent_id, local_obs, team_obs)

                    # 행동 저장
                    actions_dict[agent_id] = action

                    # [수정] ML-Agents의 reward는 "직전 행동의 결과"이므로
                    # 현재 스텝이 아니라 이전 버퍼 칸에 귀속시켜야 함
                    reward = decision_steps[agent_id].reward
                    self.episode_rewards[agent_id] += reward

                    buf = self.buffers[agent_id]
                    if len(buf) > 0:
                        buf.rewards[-1] += reward

                    # 버퍼에 추가 (보상은 다음 스텝에서 채워짐)
                    buf.add(
                        obs=local_obs,
                        team_obs=team_obs,
                        action=action,
                        log_prob=log_prob,
                        reward=0.0,
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

                if self.in_eval:
                    self.eval_steps += len(decision_steps)
                else:
                    self.total_steps += len(decision_steps)
                steps_since_update += len(decision_steps)
                episode_steps += len(decision_steps)

                # ---- 업데이트 체크 ----
                # 살아있는 에이전트 기준으로만 판단 (사망 에이전트의 짧은 버퍼가 블로킹하지 않도록)
                # 평가 중에는 학습하지 않음
                living_agents = set(self.agent_roles.keys()) - terminated_agents
                if not self.in_eval and living_agents:
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