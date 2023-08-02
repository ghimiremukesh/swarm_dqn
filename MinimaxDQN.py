import pickle

import torch
from torch import nn
from torch.nn.functional import smooth_l1_loss, mse_loss
from loss_functions import CqlLoss

import numpy as np
import matplotlib.pyplot as plt
import os

from model import DQN, StepResetLR, StepLRMargin
from collections import namedtuple
from ReplayMemory import ReplayMemory
import time
from utils import soft_update, save_model
from utils import ACTION_MAP

from torch.utils.tensorboard import SummaryWriter


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "info"))


# == helper functions :: REPLACE WITH ACTION_MAP ==
def actionIndexInt2Tuple(actionIdx, numActionList):
    """
    Transforms an action index to tuple of action indices.

    :param actionIdx: action index of the discrete action set.
    :param numActionList: number of actions in the swarms
    :return: indices of the Q-matrix (tuple of int)
    """
    numJoinAction = int(numActionList[0] * numActionList[1])
    assert (actionIdx < numJoinAction), (
        "The size of joint action set is "
        "{:d} but got index {:d}".format(numJoinAction, actionIdx)
    )
    rowIdx = actionIdx // numActionList[1]
    colIdx = actionIdx % numActionList[1]

    return (rowIdx, colIdx)


def actionIndexTuple2Int(actionIdxTuple, numActionList):
    """
    Transforms tuple of action indices to an action index.

    :param actionIdxTuple: indices of the Q-matrix (tuple of int)
    :param numActionList: number of actions of the swarms
    :return: action index of the discrete action set
    """
    rowIdx, colIdx = actionIdxTuple
    assert (
            rowIdx < numActionList[0]
    ), "The size of attacker swarm's action set is {:d} but got index {:d}".format(numActionList[0], rowIdx)
    assert (
            colIdx < numActionList[1]
    ), "The size of defender swarm's action set is {:d} but got index {:d}".format(numActionList[1], colIdx)

    actionIdx = numActionList[1] * rowIdx + colIdx

    return actionIdx


class MinimaxDQN():
    """
    Implements the deep Q-Network algorithm
    """

    def __init__(self, CONFIG, n_observations, numActionList, n_neurons, n_hidden_layers):
        """
        Initializes the environment information and neural network architecture
        :param n_obervations: # observations, aka state dim
        :param numActionList: number of actions in the action sets of the players
        :param n_hidden_layers: number of hidden layers in the neural network
        :param n_neurons: number of neurons in a hidden layer
        """
        self.CONFIG = CONFIG
        self.saved = False
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)

        self.numJoinAction = int(numActionList[0] * numActionList[1])
        self.numActionList = numActionList

        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.LR = CONFIG.LR
        self.LR_PERIOD = CONFIG.LR_PERIOD
        self.LR_DECAY = CONFIG.LR_DECAY
        self.LR_END = CONFIG.LR_END
        self.lam = CONFIG.LAM
        self.SOFT_UPDATE = CONFIG.SOFT_UPDATE
        self.HARD_UPDATE = CONFIG.HARD_UPDATE
        self.TAU = CONFIG.TAU

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.build_network(n_observations, self.numJoinAction, n_neurons, n_hidden_layers)

        self.EpsilonScheduler = StepResetLR(
            initValue=CONFIG.EPSILON,
            period=CONFIG.EPS_PERIOD,
            resetPeriod=CONFIG.EPS_RESET_PERIOD,
            decay=CONFIG.EPS_DECAY,
            endValue=CONFIG.EPS_END,
        )
        self.EPSILON = self.EpsilonScheduler.get_variable()

        self.GammaScheduler = StepLRMargin(
            initValue=CONFIG.GAMMA,
            period=CONFIG.GAMMA_PERIOD,
            decay=CONFIG.GAMMA_DECAY,
            endValue=CONFIG.GAMMA_END,
            goalValue=1.0,
        )
        self.GAMMA = self.GammaScheduler.get_variable()
        self.MAX_MODEL = CONFIG.MAX_MODEL
        self.BETA = CONFIG.BETA
        self.OMEGA = CONFIG.OMEGA

    def build_network(self, n_observations, n_actions, n_neurons, n_hidden_layers):
        """
        Builds a neural network for the Q-Network
        :param n_observations: # observations, aka state dim
        :param n_actions: number of actions in action set
        :param n_hidden_layers: number of actions in the action sets of the players
        :param n_neurons: number of neurons in a hidden layer
        """
        self.Q_network = DQN(n_observations, n_actions, n_neurons, n_hidden_layers)
        self.target_network = DQN(n_observations, n_actions, n_neurons, n_hidden_layers)

        if self.device == torch.device("cuda"):
            self.Q_network.cuda()
            self.target_network.cuda()

        self.build_optimizer()

    def build_optimizer(self):
        """
        Builds optimizer for the Q_Network and construct a scheduler for learning rate and reset counter for updates.

        """
        self.optimizer = torch.optim.Adam(
            self.Q_network.parameters(), lr=self.LR, weight_decay=1e-3
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.LR_PERIOD, gamma=self.LR_DECAY
        )
        self.max_grad_norm = 1
        self.cntUpdate = 0

    def update(self):
        """
        Updates the Q-Network using a batch of sampled replay transitions

        :return: critic loss
        """
        if len(self.memory) < self.BATCH_SIZE * 20:
            return

        # == EXPERIENCE REPLAY ==
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # # Compute a mask of non-final states and concatenate the batch elements
        # # (a final state would've been the one after which simulation ended)
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                         batch.next_state)), device=self.device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state
        #                                    if s is not None])
        # state_batch = torch.cat(batch.state)
        # action_batch = torch.cat(batch.action)
        # reward_batch = torch.cat(batch.reward)

        (non_final_mask, non_final_next_states, state, action, reward) = self.unpack_batch(batch)
        # == get Q(s, a) ==

        state_action_values = self.Q_network(state).gather(dim=1, index=action).view(-1)

        # == get a' ==
        # u', d' = argmin_u argmax_d Q_policy(s, u', d')
        # a' = tuple2idx(u', d')
        with torch.no_grad():
            num_non_final = non_final_next_states.shape[0]
            next_states_action_values = self.Q_network(non_final_next_states)
            Q_mtx = next_states_action_values.detach().reshape(
                num_non_final, self.numActionList[0], self.numActionList[1])

            # minmax values and indices
            attackerValues, colIndices = Q_mtx.max(dim=-1)
            _, rowIdx = attackerValues.min(dim=-1)
            colIdx = colIndices[np.arange(num_non_final), rowIdx]

            next_action = [actionIndexTuple2Int((r, c), self.numActionList)
                           for r, c in zip(rowIdx, colIdx)]
            next_action = (torch.LongTensor(next_action).to(self.device).view(-1, 1))

        # == get expected value ==
        next_state_values = torch.zeros(self.BATCH_SIZE).to(self.device)
        with torch.no_grad():
            # future values
            Q_expect = self.target_network(non_final_next_states)
            next_state_values[non_final_mask] = Q_expect.gather(dim=1, index=next_action).view(-1)

            # current values
            Q_curr = self.target_network(state)
            curr_state_values = Q_curr.gather(dim=1, index=action).view(-1)

        # == Compute Expected Q values == Minimax Q learning
        expected_state_action_values = (
            torch.zeros(self.BATCH_SIZE).float().to(self.device)
        )
        non_terminal = reward[non_final_mask] + self.GAMMA * next_state_values[non_final_mask]

        terminal = reward

        # # # non-terminal state : regularization
        # delta = w * (r + discount * next_state_value) + (1 - w) * current_state_value
        #                                                          - Q[state, act1, act2]

        expected_state_action_values[non_final_mask] = non_terminal

        # terminal state
        final_mask = torch.logical_not(non_final_mask)
        expected_state_action_values[final_mask] = terminal[final_mask]


        # this is with added relaxation
        # expected_state_action_values = self.OMEGA * expected_state_action_values + \
        #                                (1 - self.OMEGA) * curr_state_values

        # == Train ==
        self.Q_network.train()
        # loss = smooth_l1_loss(input=state_action_values, target=expected_state_action_values.detach())
        loss = mse_loss(input=state_action_values, target=expected_state_action_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.update_target_network()

        return loss.item()

    def unpack_batch(self, batch):
        """
        Decomposes the batch into different variables

        :param batch: Transition of batch-arrays
        :return: A tuple extracted from the elements in the batch and processed for update().
        """
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.FloatTensor(np.array([s for s in batch.next_state if s is not None])).to(
            self.device)

        state = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        action = torch.LongTensor(batch.action).to(self.device).view(-1, 1)
        reward = torch.FloatTensor(batch.reward).to(self.device)

        return (
            non_final_mask, non_final_next_states, state, action, reward
        )

    def learn(
            self, env, MAX_UPDATES=2000000, MAX_EP_STEPS=100, warmupBuffer=True,
            warmupQ=False, warmupIter=10000, runningCostThr=None,
            curUpdates=None, checkPeriod=100000, numRndTraj=200, storeModel=True,
            storeBest=True, outFolder="swarm/minimax", verbose=True
    ):
        """
        Learns the Q function given the training hyper-params
        :param env: the gym environment
        :param MAX_UPDATES: the maximum number of gradient updates
        :param MAX_EP_STEPS: the number of steps in an episode
        :param warmupBuffer: fill the replay buffer if True
        :param warmupQ: train the Q-network by l_x if True
        :param warmupIter: the number of iterations in the Q-network warmup
        :param runningCostThr: end the episode if the running cost is smaller than the threshold
        :param curUpdates: set the current number of updates
        :param checkPeriod: the period we check the performance
        :param numRndTraj: the number of random trajectories used to obtain the success ratio
        :param storeModel: store models if True
        :param storeBest: only store the best model if True
        :param outFolder: the path of the parent folder of model/
        :param verbose: print the messages if True
        :return: trainingRecords: loss for every Q-Network update
        :return: trainProgress: each entry consists of the (success, failure, unfinished) ratio of random trajectories,
        which are check periodically
        """

        #tensorboard
        model_folder = f"model_{self.BETA}"
        Writer = SummaryWriter(log_dir=os.path.join(outFolder, model_folder))

        # == Warmup Buffer ==
        startInitBuffer = time.time()
        if warmupBuffer:
            self.initBuffer(env)
        endInitBuffer = time.time()

        # == Warmup Q ==
        startInitQ = time.time()
        if warmupQ:
            pass

        # == Main Training ==
        startLearning = time.time()
        trainingRecords = []
        runningCost = 0.0
        trainProgress = []
        checkPointSucc = 0.0
        ep = 0
        if curUpdates is not None:
            self.cntUpdate = curUpdates
            print("starting from {:d} updates".format(self.cntUpdate))

        if storeModel:
            modelFolder = os.path.join(outFolder, model_folder)

        while self.cntUpdate <= MAX_UPDATES:
            state = env.reset()
            epCost = 0.0
            ep += 1
            # Rollout
            for step_num in range(MAX_EP_STEPS):
                # select action
                actionIdx, actionIdxTuple = self.select_action(state, explore=True)

                # interact with the envs
                next_state, reward, done, info = env.step(actionIdxTuple)
                epCost += reward

                # store the transition into memory
                self.store_transition(state, actionIdx, reward, next_state, info)
                state = next_state

                # check after fixed number of gradient updates
                if self.cntUpdate != 0 and self.cntUpdate % checkPeriod == 0:
                    self.Q_network.eval()
                    _, results, _, _, _, lxs = env.simulate_trajectories(
                        self.Q_network, T=MAX_EP_STEPS, num_rnd_traj=numRndTraj
                    )
                    success = np.sum(results == 1) / numRndTraj
                    # failure = np.sum(results == -1) / numRndTraj
                    unfinish = np.sum(results == 0) / numRndTraj
                    sum_rewards = np.sum(lxs)
                    trainProgress.append([success, unfinish, sum_rewards])

                    Writer.add_scalar("Success Rate (200 Trajs)", success, self.cntUpdate // checkPeriod)
                    Writer.add_scalar("Sum Rewards (200 Trajs)", sum_rewards, self.cntUpdate // checkPeriod)
                    if verbose:
                        lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
                        print("\n\nAfter [{:d}] updates:".format(self.cntUpdate))
                        print(
                            "  -e eps={:.2f}, gamma={:.6f}, lr={:.1e}.".format(
                                self.EPSILON, self.GAMMA, lr
                            )
                        )
                        print(
                            " - success/unfinished ratio: "
                            + "{:.3f}, {:.3f}".format(success, unfinish)
                        )
                        print(
                            " - sum of rewards for {:d} trajectories: {:.3f}".format(
                                numRndTraj, sum_rewards
                            )
                        )

                    if storeModel:
                        if storeBest:
                            if success > checkPointSucc:
                                checkPointSucc = success
                                self.save(self.cntUpdate, modelFolder)
                        else:
                            self.save(self.cntUpdate, modelFolder)

                # Perform one step of the optimization (on the target network)
                lossC = self.update()
                trainingRecords.append(lossC)
                self.cntUpdate += 1
                self.updateHyperParam()

                # add to tensorboard
                Writer.add_scalar("Network Loss", lossC, self.cntUpdate)


                # Terminate early
                if done:
                    break

            # Rollout Report
            runningCost = runningCost * 0.9 + epCost * 0.1
            if verbose:
                # print(
                #     "\r[{:d}-{:d}]: ".format(ep, self.cntUpdate)
                #     + "This episode gets running/episode cost = "
                #     + "({:3.2f}/{:.2f}) after {:d} steps.".format(runningCost, epCost, step_num + 1),
                #     end="",
                # )
                print(
                    "\r[{:d}-{:d}]: ".format(ep, self.cntUpdate)
                    + "This episode gets episode cost = "
                    + "({:.2f}) after {:d} steps.".format(epCost, step_num + 1),
                    end="",
                )
            # Check stopping criteria
            if runningCostThr is not None:
                if runningCost <= runningCostThr:
                    print("\n At Updates[{:3.0f}] Solved!".format(self.cntUpdate)
                          + " Running cost is now {:3.2f}!".format(runningCost)
                          )
                    env.close()
                    break

        endLearning = time.time()
        timeInitBuffer = endInitBuffer - startInitBuffer
        timeLearning = endLearning - startLearning
        self.save(self.cntUpdate, modelFolder)
        print(
            "\nInitBuffer: {:.1f}, Learning: {:.1f}".format(
                timeInitBuffer, timeLearning
            )
        )
        trainingRecords = np.array(trainingRecords)
        trainProgress = np.array(trainProgress)
        Writer.close()

        return trainingRecords, trainProgress

    def select_action(self, state, explore=False):
        """
        Selects an action given the state and conditioned on 'explore' flag.
        :param state: state of the environment
        :param explore: randomize the action by epsilon-greedy algorithm if True
        :return: action index
        :return: indices to access the Q-value matrix
        """

        if (np.random.rand() < self.EPSILON) and explore:
            actionIdx = np.random.randint(0, self.numJoinAction)
            actionIdxTuple = actionIndexInt2Tuple(actionIdx, self.numActionList)
        else:
            self.Q_network.eval()
            state = torch.from_numpy(state).float().to(self.device)
            state_action_values = self.Q_network(state)
            Q_mtx = (
                state_action_values.detach().cpu().reshape(
                    self.numActionList[0], self.numActionList[1]
                )
            )
            attackerValues, colIndices = Q_mtx.max(dim=1)
            _, rowIdx = attackerValues.min(dim=0)
            colIdx = colIndices[rowIdx]  # change this to max min Q for defender

            # # defenderValues, _ = Q_mtx.min(dim=1)
            # # _, colIdx = defenderValues.max(dim=0)
            # _, colIdx = attackerValues.max(dim=0)  # p2 maximizes the Q.

            actionIdxTuple = (rowIdx.detach().cpu().numpy().item(), colIdx.detach().cpu().numpy().item())
            actionIdx = actionIndexTuple2Int(actionIdxTuple, self.numActionList)

        return actionIdx, actionIdxTuple

    def update_target_network(self):
        """
        Updates the target network
        """
        if self.SOFT_UPDATE:
            # Soft Replace: update the target_network right after every gradient update of the Q-network by
            # target = Tau * Q_network + (1-TAU) * target
            soft_update(self.target_network, self.Q_network, self.TAU)
        elif self.cntUpdate % self.HARD_UPDATE == 0:
            # Hard Replace: copy the Q_network into the target network every HARD_UPDATE updates
            self.target_network.load_state_dict(self.Q_network.state_dict())

    def initBuffer(self, env):
        """
        Adds some transitions to the replay memory (buffer) randomly
        :param env: gym environment
        """
        count = 0
        while len(self.memory) < self.memory.capacity:
            count += 1
            print("\rWarmup Buffer [{:d}]".format(count), end="")
            state = env.reset()
            actionIdx, actionIdxTuple = self.select_action(state, explore=True)
            next_state, reward, done, info = env.step(actionIdxTuple)
            self.store_transition(state, actionIdx, reward, next_state, info)
        print("\n => Warmup Buffer Ends")

    def store_transition(self, *args):
        """
        Stores the transition into the replay buffer.
        """
        self.memory.update(Transition(*args))

    def save(self, step, logs_path):
        """
        Saves the model weights and save the configuration file in first call.

        :param step: the number of updates so far
        :param logs_path: the folder path to save the model
        """
        save_model(self.Q_network, step, logs_path, "Q", self.MAX_MODEL)
        if not self.saved:
            config_path = os.path.join(logs_path, "CONFIG.pkl")
            pickle.dump(self.CONFIG, open(config_path, "wb"))
            self.saved = True

    def restore(self, step, logs_path, verbose=True):
        """
        Restores the model weights from the given model path.

        :param step: the number of updates of the model
        :param logs_path: the folder path of the model
        :param verbose: print if True
        """
        logs_path = os.path.join(logs_path, "model", "Q-{}.pth".format(step))
        self.Q_network.load_state_dict(
            torch.load(logs_path, map_location=self.device)
        )
        if verbose:
            print("  => Restore {}".format(logs_path))

    def updateHyperParam(self):
        """
        Updates the hyper-params
        """

        lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        if (lr <= self.LR_END):
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.LR_END
        else:
            self.scheduler.step()

        self.EpsilonScheduler.step()
        self.EPSILON = self.EpsilonScheduler.get_variable()
        self.GammaScheduler.step()
        self.GAMMA = self.GammaScheduler.get_variable()
