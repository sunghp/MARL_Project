"""
spectate.py - 학습된 정책 관전 모드

학습된 체크포인트(.pt)를 불러와 화면을 켜고 정상 속도로 게임을 플레이시킨다.
학습은 하지 않고 추론만 하며, 데모용으로 결정론적(argmax) 행동을 쓴다.

사용법:
    python spectate.py --checkpoint checkpoints/mappo_ep800.pt
    python spectate.py --checkpoint checkpoints/mappo_ep800.pt --time-scale 1.0 --episodes 5

포트폴리오 영상용:
    - 여러 국면의 체크포인트를 각각 돌려서 전략 변화를 비교
    - 화면 녹화는 OBS 또는 Win+G(게임바)로
    - Unity 씬에 자유 카메라(FreeCam) 스크립트를 붙여두면 원하는 각도로 관전
"""

import argparse
import os

import numpy as np
import torch
import yaml
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

# 학습 코드 재사용 (트레이너를 수정하지 않고 그대로 임포트)
from mappo_trainer import CONFIG, MAPPOTrainer


def load_balance_params(config):
    """config/TheThing.yaml의 environment_parameters 블록을 읽어 반환.
    (트레이너 train()의 인라인 로직과 동일)"""
    path = config.get("balance_yaml")
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return loaded.get("environment_parameters", {}) or {}
    return {}


def deterministic_action(actor, obs_tensor):
    """샘플링 대신 각 브랜치의 argmax(가장 확률 높은 행동)를 선택.
    데모에선 정책의 '진짜 의도'가 깔끔하게 드러난다."""
    logits_list = actor.forward(obs_tensor)          # 브랜치별 logits 리스트
    actions = [torch.argmax(logits, dim=-1) for logits in logits_list]
    return torch.stack(actions, dim=-1)              # (batch, 브랜치수)


def spectate(checkpoint_path, time_scale=1.0, episodes=5, stochastic=False):
    config = dict(CONFIG)
    config["no_graphics"] = False        # 화면 켜기
    config["time_scale"] = time_scale    # 정상 속도 (기본 1.0)

    # 1) 트레이너 객체 생성(네트워크만 구성) + 체크포인트 로드
    trainer = MAPPOTrainer(config)
    trainer.load(checkpoint_path)
    for actor in trainer.actors.values():
        actor.eval()

    L = config["local_obs_dim"]          # 55 (앞부분 = Actor용 로컬 관측)

    # 2) Unity 환경 (화면 ON, 정상 속도)
    engine_channel = EngineConfigurationChannel()
    param_channel = EnvironmentParametersChannel()

    env = UnityEnvironment(
        file_name=config["env_path"],
        side_channels=[engine_channel, param_channel],
        base_port=5006,                  # 학습(5004)과 겹치지 않게 다른 포트
        timeout_wait=120,
        no_graphics=False,
    )
    engine_channel.set_configuration_parameters(time_scale=time_scale)

    # 밸런스 파라미터 주입 (학습 때와 동일 환경 조건)
    for key, val in load_balance_params(config).items():
        param_channel.set_float_parameter(key, float(val))

    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]

    print(f"[관전 시작] {checkpoint_path} | time_scale={time_scale} | "
          f"{'확률적' if stochastic else '결정론적(argmax)'} 행동")

    agent_roles = {}
    agent_teams = {}
    ep_done = 0

    try:
        while ep_done < episodes:
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # 종료 감지 (한 에피소드 끝)
            if len(decision_steps) == 0 and len(terminal_steps) > 0:
                ep_done += 1
                print(f"[에피소드 {ep_done}/{episodes} 종료]")
                agent_roles.clear()
                agent_teams.clear()
                env.reset()
                continue

            if len(decision_steps) == 0:
                env.step()
                continue

            # 역할 등록 + 로컬/전역 관측 분리
            all_agent_global = {}
            local_by_id = {}
            for agent_id in decision_steps.agent_id:
                agent_id = int(agent_id)
                full = decision_steps[agent_id].obs[0]        # 97차원
                local_by_id[agent_id] = full[:L]              # 앞 55 = Actor
                all_agent_global[agent_id] = full[L:]         # 뒤 42 = 전역

                if agent_id not in agent_roles:
                    role = trainer.identify_role(full)
                    agent_roles[agent_id] = role
                    agent_teams[agent_id] = trainer.get_team(role)

            # 행동 추론 (학습 없음, no_grad)
            action_array = np.zeros(
                (len(decision_steps), len(config["action_branches"])),
                dtype=np.int32,
            )
            with torch.no_grad():
                for i, agent_id in enumerate(decision_steps.agent_id):
                    agent_id = int(agent_id)
                    role = agent_roles[agent_id]
                    obs_tensor = torch.FloatTensor(local_by_id[agent_id]).unsqueeze(0).to(trainer.device)

                    if stochastic:
                        action, _ = trainer.actors[role].get_action(obs_tensor)
                    else:
                        action = deterministic_action(trainer.actors[role], obs_tensor)

                    action_array[i] = action.squeeze(0).cpu().numpy()

            env.set_actions(behavior_name, ActionTuple(discrete=action_array))
            env.step()

    except KeyboardInterrupt:
        print("\n[관전 중단]")
    finally:
        env.close()
        print("[관전 종료]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="불러올 체크포인트 .pt 경로")
    parser.add_argument("--time-scale", type=float, default=1.0, help="게임 속도 (1.0 = 정상)")
    parser.add_argument("--episodes", type=int, default=5, help="관전할 에피소드 수")
    parser.add_argument("--stochastic", action="store_true", help="argmax 대신 확률적 행동")
    args = parser.parse_args()

    spectate(args.checkpoint, args.time_scale, args.episodes, args.stochastic)