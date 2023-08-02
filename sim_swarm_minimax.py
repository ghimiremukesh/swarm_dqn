import os
import argparse
import time
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from config import CONFIG
from MinimaxDQN import MinimaxDQN
from utils import ACTION_MAP, save_obj, plot



import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import swarm

# == ARGS ==
parser = argparse.ArgumentParser()


# nn params
num_hidden_layers = 3
num_neurons = 128

# envs params
parser.add_argument(
    "-rnd", "--randomSeed", help="random seed", default=0, type=int
)

# == CONFIGURATION ==
env_name = "swarm-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("\n== Environment Information ==")
env = gym.make(
    env_name, device=device, mode="minimax")

stateDim = env.state.shape[0]
actionNum = env.action_space.n
action_list = np.arange(actionNum)

# == Agent Config ==
CONFIG = CONFIG()  # use default configs, change there if needed
maxUpdates = CONFIG.MAX_UPDATES
updatePeriod = int(maxUpdates / 20)  # for updating hyperparameters
CONFIG.GAMMA_PERIOD = updatePeriod

CONFIG.LR_PERIOD = updatePeriod

CONFIG.EPS_PERIOD = updatePeriod
CONFIG.EPS_RESET_PERIOD = maxUpdates

# path info
save_path = f"experiments/minimax/beta_{CONFIG.BETA}"
figureFolder = os.path.join(save_path, 'figure')
os.makedirs(figureFolder, exist_ok=True)
# == AGENT ==
numActionList = env.numActionList
numJoinAction = int(numActionList[0] * numActionList[1])
agent = MinimaxDQN(CONFIG, stateDim, numActionList, num_neurons, num_hidden_layers)

# skip Q warmup

print("\n== Training Information ==")
trainRecords, trainProgress = agent.learn(env, MAX_UPDATES=maxUpdates, storeBest=True)  # use default settings

trainDict = {}
trainDict['trainRecords'] = trainRecords
trainDict['trainProgress'] = trainProgress
filePath = os.path.join(save_path, 'train')

plotFigure = True
storeFigure = True


if plotFigure:
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))

    data = trainRecords
    ax = axes[0]
    ax.plot(data, 'b:')
    ax.set_xlabel('Iteration (x 1e5)', fontsize=18)
    ax.set_xticks(np.linspace(0, maxUpdates, 5))
    ax.set_xticklabels(np.linspace(0, maxUpdates, 5) / 1e5)
    ax.set_title('loss_critic', fontsize=18)
    ax.set_xlim(left=0, right=maxUpdates)
    ax.set_yscale('log')
    try:
        data = trainProgress[:, 0]
        ax = axes[1]
        x = np.arange(data.shape[0]) + 1
        ax.plot(x, data, 'b-o')
        ax.set_xlabel('Index', fontsize=18)
        ax.set_xticks(x)
        ax.set_title('Success Rate', fontsize=18)
        ax.set_xlim(left=1, right=data.shape[0])
        ax.set_ylim(0, 1.1)

        data = trainProgress[:, 2]
        ax = axes[2]
        x = np.arange(data.shape[0]) + 1
        ax.plot(x, data, 'b-o')
        ax.set_xlabel('Index', fontsize=18)
        ax.set_xticks(x)
        ax.set_title('Sum Rewards', fontsize=18)
        ax.set_xlim(left=1, right=data.shape[0])
    except:
        print("no train progress data!")

    fig.tight_layout()
    if storeFigure:
        figurePath = os.path.join(figureFolder, 'train_loss_success.png')
        fig.savefig(figurePath)
    if plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()

save_obj(trainDict, filePath)


num_trajectories = 10

ini_state = 0.25 * np.ones((8, ))
trajAttacker, trajDefender, us, ds, result, minV, info = env.simulate_one_trajectory(agent.Q_network, state=ini_state)

plot(trajAttacker, trajDefender, len(trajAttacker))

print(result)
print(trajAttacker)
print(trajDefender)