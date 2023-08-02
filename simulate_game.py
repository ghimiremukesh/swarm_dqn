import os
import argparse
import time
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from config import CONFIG
from SwarmDQN import SwarmDQN
from utils import ACTION_MAP, save_obj, plot

import swarm

# == ARGS ==
parser = argparse.ArgumentParser()


# nn params
num_hidden_layers = 3
num_neurons = 128


# == CONFIGURATION ==
env_name = "swarm-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_dir = 'swarm/minimax/model_0.25/Q-4000082.pth'

print("\n== Environment Information ==")
env = gym.make(
    env_name, device=device, mode="minimax")


if env.mode == "minimax":
    env.simulate_one_traj = env.simulate_one_trajectory_minimax
else:
    env.simulate_one_traj = env.simulate_one_trajectory

stateDim = env.state.shape[0]
actionNum = env.action_space.n
action_list = np.arange(actionNum)

# == Agent Config ==
CONFIG = CONFIG()  # use default configs, change there if needed
# == AGENT ==
numActionList = env.numActionList

agent = SwarmDQN(CONFIG, stateDim, numActionList, num_neurons, num_hidden_layers)

agent.Q_network.load_state_dict(torch.load(model_dir, map_location=device))
agent.Q_network.eval()

# skip Q warmup,

num_trajectories = 1

ini_state = 0.25 * np.ones((8, ))
np.random.seed(10)
trajAttacker, trajDefender, us, ds, result, minV, info = env.simulate_one_traj(agent.Q_network, state=ini_state)

try:
    plot(trajAttacker, trajDefender, len(trajAttacker))
except:
    print("Too huge to plot, select few time-steps along the way to plot")

print(result)
# print(trajAttacker)
# print(trajDefender)
print(us)


print(ds)
# print(minV)