"""
PPO+SCPR Ablation Baseline
===========================
Runs 5 seeds of PPO with state-conditioned greedy top-k prescreening (no GA crossover/mutation).
This is the critical ablation to isolate the GA contribution vs. adaptive fitness ranking alone.

Usage:
    python run_scpr_ablation.py

Results saved to: rewards_SCPR_seed{N}.txt
Final mean+/-std saved to: rewards_SCPR_summary.txt
"""
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

# Monkey-patch parameters to enable SCPR mode
from production.envs import initialize_env as _init_env
_original_define = _init_env.define_production_parameters

def _scpr_define_production_parameters(env, episode):
    params = _original_define(env=env, episode=episode)
    params['USE_GA'] = True        # enable GA prescreening pathway
    params['USE_SCPR_ONLY'] = True  # bypass crossover/mutation, greedy top-k only
    return params

_init_env.define_production_parameters = _scpr_define_production_parameters

# Import Transport after patching so class-level reset works
from production.envs.transport import Transport
from production.envs.production_env import ProductionEnv

with open('config/ppo1.json', 'r') as f:
    config = json.load(f)

timesteps = 320
episodes = 2500
total_timesteps = timesteps * episodes
N_SEEDS = 5
all_seed_rewards = []


class RewardCallback(BaseCallback):
    def __init__(self, total_episodes, max_timesteps_per_episode, seed_id, verbose=0):
        super().__init__(verbose)
        self.total_episodes = total_episodes
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.seed_id = seed_id
        self.current_episode_reward = 0
        self.timesteps_in_episode = 0
        self.episode_rewards = []
        self.pbar = tqdm(total=self.total_episodes, desc=f"Seed {seed_id}", unit="ep")

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.timesteps_in_episode += 1
        if self.timesteps_in_episode >= self.max_timesteps_per_episode:
            avg_reward = self.current_episode_reward / self.max_timesteps_per_episode
            self.episode_rewards.append(avg_reward)
            with open(f"rewards_SCPR_seed{self.seed_id}.txt", "a") as f:
                f.write(f"{avg_reward}\n")
            self.current_episode_reward = 0
            self.timesteps_in_episode = 0
            self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()


print("=== PPO+SCPR Ablation (greedy top-k, no GA crossover/mutation) ===")

for seed in range(N_SEEDS):
    print(f"\n--- Seed {seed+1}/{N_SEEDS} ---")

    # Reset Transport class-level state before each seed to prevent cross-seed contamination
    Transport.all_transp_orders = []
    Transport.agents_waiting_for_action = []
    Transport.state_vector = []

    env = ProductionEnv(max_episode_timesteps=timesteps)
    config_copy = dict(config)
    policy = config_copy.pop('policy')
    model = PPO(policy, env, seed=seed, **config_copy)

    callback = RewardCallback(
        total_episodes=episodes,
        max_timesteps_per_episode=timesteps,
        seed_id=seed + 1
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    env.close()

    seed_rewards = callback.episode_rewards
    all_seed_rewards.append(seed_rewards)
    partial_mean = np.mean(seed_rewards[-500:]) if len(seed_rewards) >= 500 else np.mean(seed_rewards)
    print(f"  Seed {seed+1} converged mean (last 500 ep): {partial_mean:.3f}")

# Summary statistics (last 500 converged episodes per seed)
final_means = [
    np.mean(r[-500:]) if len(r) >= 500 else np.mean(r)
    for r in all_seed_rewards
]
overall_mean = np.mean(final_means)
overall_std = np.std(final_means)

print(f"\n=== SCPR Summary (last 500 episodes per seed) ===")
for i, m in enumerate(final_means):
    print(f"  Seed {i+1}: {m:.3f}")
print(f"  Mean +/- Std: {overall_mean:.3f} +/- {overall_std:.3f}")

with open("rewards_SCPR_summary.txt", "w") as f:
    f.write("PPO+SCPR Ablation Results\n")
    f.write(f"Mean: {overall_mean:.4f}\n")
    f.write(f"Std:  {overall_std:.4f}\n")
    for i, m in enumerate(final_means):
        f.write(f"Seed{i+1}: {m:.4f}\n")

print("\nDone. Compare SCPR mean vs HGPPO (153.31) to quantify GA evolutionary operator contribution.")
