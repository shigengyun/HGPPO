import json
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from production.envs.production_env import ProductionEnv

# Edit this path to point to your trained model
model_path = "model_history2/model_20251118-120100.zip"
model = PPO.load(model_path)

timesteps = 320
episodes = 100
env = ProductionEnv(max_episode_timesteps=timesteps)
check_env(env)

rewards = []
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title('Reward Fluctuations Over Episodes')

for episode in range(episodes):
    obs, info = env.reset()
    episode_reward = 0
    for step in range(timesteps):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        if done:
            break
    episode_reward = episode_reward / timesteps
    rewards.append(episode_reward)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    ax.clear()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward Fluctuations Over Episodes (Evaluation)')
    ax.plot(rewards, label='Total Reward')
    with open("rewards_HGPPO1.txt", "w") as file:
        for rw in rewards:
            file.write(f"{rw}\n")
    ax.legend()

    plt.draw()
    plt.pause(0.01)

plt.ioff()
plt.show()
