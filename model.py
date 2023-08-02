import torch
from torch import nn
from collections import OrderedDict
import abc

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)

class DQN(nn.Module):
    """
    Deep Q-Network Architecture
    """

    def __init__(self, n_observations, n_actions, n_neurons, n_hidden_layers):
        super(DQN, self).__init__()
        self.nl = nn.LeakyReLU(inplace=True)
        self.final_nl = nn.LeakyReLU(inplace=True)
        self.net = []
        self.net.append(nn.Sequential(nn.Linear(n_observations, n_neurons), self.nl))
        for i in range(n_hidden_layers):
            self.net.append(nn.Sequential(
                nn.Linear(n_neurons, n_neurons), self.nl
            ))

        self.net.append(nn.Sequential(
            nn.Linear(n_neurons, n_actions), self.final_nl
        ))

        self.net = nn.Sequential(*self.net)

        # initialize weights xavier normal
        self.net.apply(init_weights)

    def forward(self, x, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(x)

        return output


# == Scheduler ==
class _scheduler(abc.ABC):
    """
  The parent class for schedulers. It implements some basic functions that will
  be used in all scheduler.
  """

    def __init__(self, last_epoch=-1, verbose=False):
        """Initializes the scheduler with the index of last epoch.
    """
        self.cnt = last_epoch
        self.verbose = verbose
        self.variable = None
        self.step()

    def step(self):
        """Updates the index of the last epoch and the variable.
    """
        self.cnt += 1
        value = self.get_value()
        self.variable = value

    @abc.abstractmethod
    def get_value(self):
        raise NotImplementedError

    def get_variable(self):
        """Returns the variable.
    """
        return self.variable


class StepLR(_scheduler):
    """This scheduler will decay to end value periodically.
  """

    def __init__(
            self, initValue, period, decay=0.1, endValue=0., last_epoch=-1,
            verbose=False
    ):
        """Initializes an object of the scheduler with the specified attributes.

    Args:
        initValue (float): initial value of the variable.
        period (int): the period to update the variable.
        decay (float, optional): the amount by which the variable decays.
            Defaults to 0.1.
        endValue (float, optional): the target value to decay to.
            Defaults to 0.
        last_epoch (int, optional): the index of the last epoch.
            Defaults to -1.
        verbose (bool, optional): print messages if True. Defaults to False.
    """
        self.initValue = initValue
        self.period = period
        self.decay = decay
        self.endValue = endValue
        super(StepLR, self).__init__(last_epoch, verbose)

    def get_value(self):
        """Returns the value of the variable.
    """
        if self.cnt == -1:
            return self.initValue

        numDecay = int(self.cnt / self.period)
        tmpValue = self.initValue * (self.decay ** numDecay)
        if self.endValue is not None and tmpValue <= self.endValue:
            return self.endValue
        return tmpValue


class StepLRMargin(_scheduler):

    def __init__(
            self, initValue, period, goalValue, decay=0.1, endValue=1, last_epoch=-1,
            verbose=False
    ):
        """Initializes an object of the scheduler with the specified attributes.

    Args:
        initValue (float): initial value of the variable.
        period (int): the period to update the variable.
        goalValue (float):the target value to anneal to.
        decay (float, optional): the amount by which the margin between the
            variable and the goal value decays. Defaults to 0.1.
        endValue (float, optional): the maximum value of the variable.
            Defaults to 1.
        last_epoch (int, optional): the index of the last epoch.
            Defaults to -1.
        verbose (bool, optional): print messages if True. Defaults to False.
    """
        self.initValue = initValue
        self.period = period
        self.decay = decay
        self.endValue = endValue
        self.goalValue = goalValue
        super(StepLRMargin, self).__init__(last_epoch, verbose)

    def get_value(self):
        """Returns the value of the variable.
    """
        if self.cnt == -1:
            return self.initValue

        numDecay = int(self.cnt / self.period)
        tmpValue = self.goalValue - (self.goalValue
                                     - self.initValue) * (self.decay ** numDecay)
        if self.endValue is not None and tmpValue >= self.endValue:
            return self.endValue
        return tmpValue


class StepResetLR(_scheduler):

    def __init__(
            self, initValue, period, resetPeriod, decay=0.1, endValue=0,
            last_epoch=-1, verbose=False
    ):
        """Initializes an object of the scheduler with the specified attributes.

    Args:
        initValue (float): initial value of the variable.
        period (int): the period to update the variable.
        resetPeriod (int): the period to reset the variable to its initial
            value.
        decay (float, optional): the amount by which the variable decays.
            Defaults to 0.1.
        endValue (float, optional): the target value to decay to.
            Defaults to 0.
        last_epoch (int, optional): the index of the last epoch.
            Defaults to -1.
        verbose (bool, optional): print messages if True. Defaults to False.
    """
        self.initValue = initValue
        self.period = period
        self.decay = decay
        self.endValue = endValue
        self.resetPeriod = resetPeriod
        super(StepResetLR, self).__init__(last_epoch, verbose)

    def get_value(self):
        """Returns the value of the variable.
    """
        if self.cnt == -1:
            return self.initValue

        numDecay = int(self.cnt / self.period)
        tmpValue = self.initValue * (self.decay ** numDecay)
        if self.endValue is not None and tmpValue <= self.endValue:
            return self.endValue
        return tmpValue

    def step(self):
        """
    Updates the index of the last epoch and the variable. It overrides the same
    function in the parent class.
    """
        self.cnt += 1
        value = self.get_value()
        self.variable = value
        if (self.cnt + 1) % self.resetPeriod == 0:
            self.cnt = -1
