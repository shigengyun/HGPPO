import matplotlib.pyplot as plt
import numpy as np

with open("rewards_a2c.txt", "r") as f:
    a2c_data = [float(line.strip()) for line in f.readlines()]
with open("rewards_ppo.txt", "r") as f:
    ppo_data = [float(line.strip()) for line in f.readlines()]
with open("rewards_dqn.txt", "r") as f:
    dqn_data = [float(line.strip()) for line in f.readlines()]
with open("rewards_EMPTY.txt", "r") as f:
    EMPTY_data = [float(line.strip()) for line in f.readlines()]
with open("rewards_FIFO.txt", "r") as f:
    FIFO_data = [float(line.strip()) for line in f.readlines()]
with open("rewards_HGPPO.txt", "r") as f:
    HGPPO_data = [float(line.strip()) for line in f.readlines()]


def smooth_curve(data, weight=0.93, interpolation_points=1):
    # Step 1: interpolate to increase density
    dense_data = []
    for i in range(len(data) - 1):
        start = data[i]
        end = data[i + 1]
        for j in range(interpolation_points):
            alpha = j / interpolation_points
            interpolated = (1 - alpha) * start + alpha * end
            dense_data.append(interpolated)
    dense_data.append(data[-1])
    # Step 2: exponential smoothing
    smoothed = []
    last = dense_data[0]
    for point in dense_data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


a2c_data_smooth = smooth_curve(a2c_data)
dqn_data_smooth = smooth_curve(dqn_data)
ppo_data_smooth = smooth_curve(ppo_data)
FIFO_data_smooth = smooth_curve(FIFO_data)
HGPPO_data_smooth = smooth_curve(HGPPO_data)
EMPTY_data_smooth = smooth_curve(EMPTY_data)

plt.plot(HGPPO_data_smooth, label="HG_PPO", alpha=0.95)
plt.plot(ppo_data_smooth, label="PPO", alpha=0.95)

plt.ylabel("Total Reward")
plt.title("Total Reward in Episode")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(15))
plt.grid(True, alpha=0.8)
plt.legend()
plt.show()
