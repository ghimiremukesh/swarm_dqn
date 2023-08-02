"""
This module implements an environment for the two-player zero sum swarm game where the swarms are modeled using
kolmogorov dynamical system. Swarms take high-level control action that allow them to move across the region defined by
a directed graph.
"""

import gym.spaces
import numpy as np
import gym
import torch
import random

from .swarm_dyn import SwarmDyn


class SwarmEnv(gym.Env):
    """
    A gym environment considering an attack-defend game between two identical swarms
    """

    def __init__(self, device, mode=None):
        """
        Initializes the environment
        :param device: device type (cpu or cuda)
        :param mode: algorithm type (minimax, reach-avoid, etc)
        """
        # Set random seed
        self.set_seed(0)

        self.mode = mode

        # Gym vars
        self.numActionList = [16, 16]
        self.action_space = gym.spaces.MultiBinary(8)  # each agent has 4 actions -- 1 in each edge
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8,))

        # Target set param
        self.target_diff = 0.25

        # Swarm params
        self.time_step = 0.05
        self.attacker = SwarmDyn()  # init attacker
        self.defender = SwarmDyn()  # init defender

        # Internal state
        self.state = np.zeros(8)

        # Cost params
        self.penalty = 1.  # penalty at each time-step
        if self.mode == "minimax":
            self.reward = -100. # when target is reached award -100
        else:
            self.reward = -1.
        self.device = device

    def reset(self, start=None):
        """
        Resets the state of the environment.
        :param start: state to reset the environment to. If None, pick randomly.
        :return: state that the envs has been reset to.
        """
        if start is not None:
            stateAttacker = self.attacker.reset(start=start[:4])
            stateDefender = self.defender.reset(start=start[4:])
        else:
            state = self.sample_random_state(sample_inside_tar=False)
            stateAttacker = state[:4]
            stateDefender = state[4:]
            self.attacker.state = stateAttacker
            self.defender.state = stateDefender
            # stateAttacker = self.attacker.reset()
            # stateDefender = self.defender.reset()
        self.state = np.concatenate((stateAttacker, stateDefender), axis=0)

        return np.copy(self.state)

    def sample_random_state(self, sample_inside_tar=False):
        """
        Picks the state at random.
        :param sample_inside_tar: consider sampling inside the target if True
        :return: sampled initial state
        """

        flag = True
        while flag:
            stateAttacker = self.attacker.sample_random_state()
            stateDefender = self.defender.sample_random_state()

            attacker_l_x = self.attacker.target_margin(stateAttacker, stateDefender)
            defender_l_x = self.defender.target_margin(stateDefender, stateAttacker)

            l_x = min(attacker_l_x, defender_l_x)

            if (not sample_inside_tar) and (l_x < 0):
                flag = True
            else:
                flag = False

        return np.concatenate((stateAttacker, stateDefender), axis=0)

    # == Dynamics Functions ==
    def step(self, action):
        """
        Evolves the environment one step forward under given action.

        :param action: contains the list of action indexes in the attacker and defender's action set, respectively.
        :return: next state, cost, terminated_or_not(bool), target margin
        """

        state_tmp = np.concatenate((self.attacker.state, self.defender.state), axis=0)
        distance = np.linalg.norm(self.state - state_tmp)
        assert distance < 1e-8, ("There is a mismatch between the envs state" +
                                 "and swarm state: {:.2e}".format(distance))

        stateAttacker, doneAttacker = self.attacker.step(action[0], self.state[4:])
        stateDefender, doneDefender = self.defender.step(action[1], self.state[:4])

        self.state = np.concatenate((stateAttacker, stateDefender), axis=0)
        attacker_l_x, defender_l_x = self.target_margin(self.state)
        success = attacker_l_x < 0
        # fail = defender_l_x < 0

        # cost
        # if fail:
        #     cost = self.penalty
        # elif success:
        #     cost = self.reward
        # else:
        #     cost = self.penalty
        if self.mode == "minimax":
            if success:
                cost = self.reward
            else:
                cost = self.penalty
        else:
            cost = attacker_l_x  # cost is 1

        # done signal # no fail.
        done = success
        # info = {"attacker_distance_to_target": attacker_l_x, "defender_distance_to_target": defender_l_x}
        info = {"l_x": attacker_l_x}

        return np.copy(self.state), cost, done, info

    def set_seed(self, seed):
        """
        Sets the seed for 'numpy', and 'PyTorch' packages.
        :param seed: seed value
        """
        self.seed_val = seed
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)
        random.seed(self.seed_val)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def target_margin(self, s):
        """
        Computes the margin between the swarms and the target set.
        :param s: current state.
        :return: distance between the current state and target set.
        """
        attacker_l_x = self.attacker.target_margin(s[:4], s[4:])
        defender_l_x = self.defender.target_margin(s[4:], s[:4])

        return attacker_l_x, defender_l_x

    def get_warmup_examples(self, num_warmup_samples=100):
        """
        Gets the sample to initialize the Q-Network.

        :param num_warmup_samples: # warmup samples.
        :return: sampled states, heuristic values -- distance_to_target
        """

        stateAttacker = np.random.dirichlet(np.ones(4), size=num_warmup_samples)
        stateDefender = np.random.dirichlet(np.ones(4), size=num_warmup_samples)

        states = np.concatenate((stateAttacker, stateDefender), axis=1)

        heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))

        for i in range(num_warmup_samples):
            state = states[i]
            l_x, _ = self.target_margin(state)
            heuristic_v[i, :] = l_x

        return states, heuristic_v

    # Probably not necessary!
    def get_value(self, q_func, stateAttacker, stateDefender, verbose=False):
        """
        Gets the state values given the Q-network.
        :param q_func: agent's Q-network
        :param stateAttacker: state of the attacker across num_regions
        :param stateDefender: state of the defender across num_regions
        :param verbose: print if True
        :return: values
        """
        if verbose:
            print(f"Getting values with attacker's state: "
                  f"{stateAttacker[0], stateAttacker[1], stateAttacker[2], stateAttacker[3]}," f" and defender's state:"
                  f" {stateDefender[0], stateDefender[1], stateDefender[2], stateDefender[3]} ")

        pass

    def simulate_trajectories(self, q_func, T=10, num_rnd_traj=None, states=None):
        """
        Simulates the trajectories
        :param q_func: agent's Q-Network
        :param T: maximum length of trajectory
        :param num_rnd_traj: # states. Defaults to None
        :param states: initial states, if provided
        :return: trajectories, outcome, minVal
        """

        # set seed for reproducibility and fair comparison
        np.random.seed(42)


        assert ((num_rnd_traj is None and states is not None)
                or (num_rnd_traj is not None and states is None)
                or (len(states) == num_rnd_traj))

        trajectories = []

        if self.mode == "minimax":
            self.simulate_one_traj = self.simulate_one_trajectory_minimax
        else:
            self.simulate_one_traj = self.simulate_one_trajectory
        if states is None:
            results = np.empty(shape=(num_rnd_traj, ), dtype=int)
            minVs = np.empty(shape=(num_rnd_traj, ), dtype=float)
            Us = np.empty(shape=(num_rnd_traj, ), dtype=object)
            Ds = np.empty(shape=(num_rnd_traj, ), dtype=object)
            lxlists = np.empty(shape=(num_rnd_traj, ), dtype=float)
            for idx in range(num_rnd_traj):
                trajAttacker, trajDefender, us, ds, result, minV, lxlist = self.simulate_one_traj(
                    q_func, T=T
                )

                trajectories.append((trajAttacker, trajDefender))
                results[idx] = result
                minVs[idx] = minV
                lxlists[idx] = sum(lxlist)
                Us[idx] = [''.join(map(str, row)) for row in us]
                Ds[idx] = [''.join(map(str, row)) for row in us]
        else:
            results = np.empty(shape=(len(states), ), dtype=int)
            minVs = np.empty(shape=(len(states), ), dtype=float)
            Us = np.empty(shape=(len(states), ), dtype=int)
            Ds = np.empty(shape=(len(states), ), dtype=int)
            lxlists = np.empty(shape=(num_rnd_traj,), dtype=object)
            for idx, state in enumerate(states):
                trajAttacker, trajDefender, us, ds, result, minV, lxlist = self.simulate_one_traj(
                    q_func, T=T, state=state
                )
                trajectories.append((trajAttacker, trajDefender))
                results[idx] = result
                minVs[idx] = minV
                lxlists[idx] = lxlist
                Us[idx] = us
                Ds[idx] = ds

        return trajectories, results, minVs, Us, Ds, lxlists

    def simulate_one_trajectory(self, q_func, T=250, state=None):
        """
        Simulates the trajectory give the state or randomly initialized.

        :param q_func: agent's Q-network
        :param T: the maximum length of the trajectory
        :param state: initial state, if provided
        :return: state, done, value of the trajectory
        """
        # reset
        if state is None:
            state = self.sample_random_state(sample_inside_tar=False)
            stateAttacker = state[:4]
            stateDefender = state[4:]
            # self.attacker.state = stateAttacker
            # self.defender.state = stateDefender
        else:
            stateAttacker = state[:4]
            stateDefender = state[4:]

        trajAttacker = []
        trajDefender = []
        us = []
        ds = []
        result = -1  # not finished

        valueList = []
        lxList = []

        for t in range(T):
            trajAttacker.append(stateAttacker)
            trajDefender.append(stateDefender)
            state = np.concatenate((stateAttacker, stateDefender), axis=0)

            l_x, defender_l_x = self.target_margin(state)
            # defender_l_x = self.defender.target_margin(stateDefender, stateAttacker)

            done = l_x < 0  # only check if attacker has captured, defender doesn't need to capture.

            if t == 0:
                current = l_x
                minV = current
            else:
                current = l_x
                minV = min(current, l_x)

            valueList.append(minV)
            lxList.append(l_x)

            result = done
            if done:
                break
                # if l_x < 0:
                #     # print("Attacker reached target before defender! Attacker Won!")
                #     result = 1
                # elif defender_l_x < 0:
                #     result = -1
                #     # print("Defender reached target before attacker! Attacker Lost!")
                # break

            # == Dynamics ==
            stateTensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                state_action_values = q_func(stateTensor)
            Q_mtx = state_action_values.reshape(self.numActionList[0], self.numActionList[1])

            attackerValues, colIndices = Q_mtx.max(dim=1)
            _, rowIdx = attackerValues.min(dim=0)
            # colIdx = colIndices[rowIdx]

            # defenderValues, _ = Q_mtx.min(dim=1)
            # _, colIdx = defenderValues.max(dim=0)
            # _, colIdx = attackerValues.max(dim=0)

            _, colIdx = Q_mtx[rowIdx, :].max(dim=0)  # player 1 plays first

            # Controls
            u = self.attacker.discrete_controls[rowIdx.item()]
            stateAttacker = self.attacker.integrate_forward(stateAttacker, u)

            d = self.defender.discrete_controls[colIdx.item()]
            stateDefender = self.defender.integrate_forward(stateDefender, d)

            us.append(u)
            ds.append(d)

        trajAttacker = np.array(trajAttacker)
        # info = {'valueList': valueList, 'lxList': lxList}

        return trajAttacker, trajDefender, us, ds, result, minV, lxList

    def simulate_one_trajectory_minimax(self, q_func, T=250, state=None):
        """
        Simulates the trajectory give the state or randomly initialized.

        :param q_func: agent's Q-network
        :param T: the maximum length of the trajectory
        :param state: initial state, if provided
        :return: state, done, value of the trajectory
        """
        # reset
        if state is None:
            state = self.sample_random_state(sample_inside_tar=False)
            stateAttacker = state[:4]
            stateDefender = state[4:]
            # self.attacker.state = stateAttacker
            # self.defender.state = stateDefender
        else:
            stateAttacker = state[:4]
            stateDefender = state[4:]

        trajAttacker = []
        trajDefender = []
        us = []
        ds = []
        result = -1  # not finished

        # valueList = []
        rewardList = []

        for t in range(T):
            trajAttacker.append(stateAttacker)
            trajDefender.append(stateDefender)
            state = np.concatenate((stateAttacker, stateDefender), axis=0)

            l_x, defender_l_x = self.target_margin(state)
            # defender_l_x = self.defender.target_margin(stateDefender, stateAttacker)

            done = l_x < 0  # only check if attacker has captured, defender doesn't need to capture.

            if done:
                reward = -100
            else:
                reward = 1

            # if t == 0:
            #     cost = reward
            # else:
            #     current = l_x
            #     minV = min(current, l_x)
            #
            # valueList.append(minV)
            rewardList.append(reward)

            result = done
            if done:
                break
                # if l_x < 0:
                #     # print("Attacker reached target before defender! Attacker Won!")
                #     result = 1
                # elif defender_l_x < 0:
                #     result = -1
                #     # print("Defender reached target before attacker! Attacker Lost!")
                # break

            # == Dynamics ==
            stateTensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                state_action_values = q_func(stateTensor)
            Q_mtx = state_action_values.reshape(self.numActionList[0], self.numActionList[1])

            if self.mode == "minimax":
                attackerValues, colIndices = Q_mtx.max(dim=1)
                _, rowIdx = attackerValues.min(dim=0)
                colIdx = colIndices[rowIdx]

            else:
                attackerValues, colIndices = Q_mtx.max(dim=1)
                _, rowIdx = attackerValues.min(dim=0)
                # colIdx = colIndices[rowIdx]

                # defenderValues, _ = Q_mtx.min(dim=1)
                # _, colIdx = defenderValues.max(dim=0)
                # _, colIdx = attackerValues.max(dim=0)

                _, colIdx = Q_mtx[rowIdx, :].max(dim=0)  # player 1 plays first

            # Controls
            u = self.attacker.discrete_controls[rowIdx.item()]
            stateAttacker = self.attacker.integrate_forward(stateAttacker, u)

            d = self.defender.discrete_controls[colIdx.item()]
            stateDefender = self.defender.integrate_forward(stateDefender, d)

            us.append(u)
            ds.append(d)

        trajAttacker = np.array(trajAttacker)
        # info = {'valueList': valueList, 'lxList': lxList}

        return trajAttacker, trajDefender, us, ds, result, None, rewardList  # None to keep same format





