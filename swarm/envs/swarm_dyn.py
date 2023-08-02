"""
Parent class for the Swarm environment
"""

import numpy as np
from utils import ACTION_MAP


class SwarmDyn(object):
    """
    This base class implements a dynamical system representing swarms.
    A target set is given that the attacker swarm tries to reach.
    """

    def __init__(self):
        """
        Initializes the environment. Episode terminates when either the attacker or the defender swarm captures a
        region.
        """
        self.num_regions = 4

        # Environment params
        self.time_step = 0.05
        self.alive = True

        # Swarm control params
        self.u_max = 1
        self.discrete_controls = ACTION_MAP

        # Target param
        self.target_diff = 0.25  # threshold for capture

        # Internal state
        self.state = np.zeros(self.num_regions)

        # Set random seed
        self.seed_val = 0
        np.random.seed(self.seed_val)

    def reset(self, start=None):
        """Resets the environment.

        Args:
            start (np.ndarray, optional): the state to reset the swarms to. If None, the states will reset to uniform
            density over all regions.

        Returns:
            np.ndarray: the state that the swarm has been reset to.
        """

        if start is not None:
            self.state = start
        else:
            self.state = np.array([0.25, 0.25, 0.25, 0.25])

        return np.copy(self.state)

    def sample_random_state(self):
        """Picks the state at random

        Returns:
             np.ndarray: the sampled initial state
        """

        # set seed for reporducibility and fair comparison
        # np.random.seed(10)
        rnd_state = np.random.dirichlet(np.ones(self.num_regions))
        x1, x2, x3, x4 = rnd_state

        return x1, x2, x3, x4

    # == Dynamics ==
    def step(self, action, opp_state):
        """
        Evolves the environment one step forward given an action.

        :param action: the index of the action in the action set.
        :param opp_state: state of the opponent
        :return:
                np.ndarray: next state.
                bool: True if the episode is terminated
        """
        l_x_curr = self.target_margin(self.state, opp_state)

        u = self.discrete_controls[action]
        state = self.integrate_forward(self.state, u)
        self.state = state

        success = l_x_curr < 0
        done = success

        if done:
            self.alive = False

        return np.copy(self.state), done

    def integrate_forward(self, state, u):
        """
        Integrates the dynamics forward by one step
        :param state: state of the swarm
        :param u: the actions for all edges
        :return: next state
        """

        x1, x2, x3, x4 = state
        u1, u2, u3, u4 = u

        x1 += self.time_step * (u4 * x4 - u1 * x1)
        x2 += self.time_step * (u1 * x1 - u2 * x2)
        x3 += self.time_step * (u2 * x2 - u3 * x3)
        x4 += self.time_step * (u3 * x3 - u4 * x4)

        state = np.array([x1, x2, x3, x4], dtype=np.float32)
        return state

    def target_margin(self, state, opp_state):
        """
        Computes the margin between the state and the target set.

        :param state: the state of the current player
        :param opp_state: the state of the opposite player
        :return: float: negative numbers indicate reaching the target.
        """
        me = np.array(state)
        you = np.array(opp_state)
        if self.target_diff is not None:
            target_margin = self.target_diff - np.max(me - you)
            return target_margin
        else:
            return None




