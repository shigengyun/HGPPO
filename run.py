import os
import time
import json
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, A2C, DQN, DDPG, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm
from production.envs.production_env import ProductionEnv

with open('config/ppo1.json', 'r') as f:
    config = json.load(f)

timesteps = 320
episodes = 2500
total_timesteps = timesteps * episodes

env = ProductionEnv(max_episode_timesteps=timesteps)
check_env(env)

algorithm = "PPO"
policy = config.pop('policy')
if algorithm == "PPO":
    model = PPO(policy, env, **config)
elif algorithm == "DQN":
    model = DQN(policy, env, **config)
elif algorithm == "A2C":
    model = A2C(policy, env, **config)
elif algorithm == "DDPG":
    model = DDPG(policy, env, **config)
elif algorithm == "TD3":
    model = TD3(policy, env, **config)

rewards = []
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title('Reward Fluctuations Over Episodes')


class RewardLossCallback(BaseCallback):
    def __init__(self, total_episodes, max_timesteps_per_episode, verbose=0):
        super().__init__(verbose)
        self.total_episodes = total_episodes
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.current_episode_reward = 0
        self.timesteps_in_episode = 0
        self.episode_rewards = []
        self.losses = []
        self.pbar = tqdm(total=self.total_episodes, desc="Training Progress", unit="episode")

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.timesteps_in_episode += 1

        if self.timesteps_in_episode >= self.max_timesteps_per_episode:
            avg_reward = self.current_episode_reward / self.max_timesteps_per_episode
            self.episode_rewards.append(avg_reward)
            with open("rewards_HGPPO.txt", "a") as f:
                f.write(f"{avg_reward}\n")
            self.current_episode_reward = 0
            self.timesteps_in_episode = 0

            ax.clear()
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
            ax.set_title('Reward Fluctuations Over Episodes')
            ax.plot(self.episode_rewards)
            plt.draw()
            plt.pause(0.01)
            self.pbar.update(1)

        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


reward_callback = RewardLossCallback(total_episodes=episodes, max_timesteps_per_episode=timesteps)

model.learn(total_timesteps=total_timesteps, callback=reward_callback)

plt.ioff()
plt.show(block=True)

# Save trained model
os.makedirs("model_history2", exist_ok=True)
timestamp = time.strftime("%Y%m%d-%H%M%S")
model.save(f"model_history2/model_{timestamp}")

env.statistics.update({'time_end': env.env.now})
