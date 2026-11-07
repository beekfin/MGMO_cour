"""Проверка сред, фиксированной сетки и вычислений модели без обучения."""
import argparse
import json
from pathlib import Path
import platform
import time

import gymnasium as gym
import numpy as np
import torch

from .actions import uniform_grid, environment_action
from .model import MuZeroModel


def run_environment(name, episodes, seed):
    env = gym.make(name)
    grid = uniform_grid(float(env.action_space.low[0]), float(env.action_space.high[0]))
    rng = np.random.default_rng(seed)
    records = []
    start = time.perf_counter()
    try:
        for episode in range(episodes):
            obs,_ = env.reset(seed=seed+episode)
            total, steps = 0.0, 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = environment_action(grid, int(rng.integers(len(grid))))
                if not env.action_space.contains(action):
                    raise ValueError("Действие вне пространства среды")
                obs,reward,terminated,truncated,_ = env.step(action)
                if not np.isfinite(obs).all() or not np.isfinite(reward):
                    raise ValueError("Неконечное наблюдение или награда")
                total += float(reward)
                steps += 1
            records.append({"seed":seed+episode,"return":total,"steps":steps,
                            "terminated":bool(terminated),"truncated":bool(truncated)})
        environment_seconds=time.perf_counter()-start
        model = MuZeroModel(len(obs),len(grid),action_scale=max(abs(grid)))
        x=torch.from_numpy(obs).unsqueeze(0)
        action=torch.from_numpy(environment_action(grid,2)).unsqueeze(0)
        with torch.no_grad():
            for _ in range(10):
                first=model.initial_inference(x)
                next_prediction=model.recurrent_inference(first.hidden,action)
            before=time.perf_counter()
            for _ in range(1000):
                first=model.initial_inference(x)
                next_prediction=model.recurrent_inference(first.hidden,action)
            model_seconds=time.perf_counter()-before
        return {"environment":name,"observation_shape":list(obs.shape),
                "action_shape":list(env.action_space.shape),"grid":grid.tolist(),
                "episode_limit":env.spec.max_episode_steps,"episodes":records,
                "environment_seconds":environment_seconds,
                "model_pairs":1000,"model_pair_mean_seconds":model_seconds/1000,
                "hidden_shape":list(first.hidden.shape),
                "policy_shape":list(first.policy_logits.shape),
                "purpose":"random_grid_and_untrained_model_check"}
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes",type=int,default=3)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--out",type=Path,default=Path("artifacts/muzero_stage1/report.json"))
    args=parser.parse_args()
    if args.episodes < 1:
        parser.error("Нужно не меньше одного эпизода")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    report={"python":platform.python_version(),"gymnasium":gym.__version__,
            "numpy":np.__version__,"torch":str(torch.__version__),
            "device":"cpu","threads":1,"trained":False,"mcts_implemented":False,
            "environments":[run_environment(name,args.episodes,args.seed)
                            for name in ("Pendulum-v1","MountainCarContinuous-v0")]}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
