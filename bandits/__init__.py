"""Bandit algorithms used in FishEconomy experiments."""

from .base import BaseBandit
from .epsilon_greedy import EpsilonGreedyBandit
from .pac import PACSuccessiveEliminationBandit
from .softmax import SoftmaxBandit
from .ucb import UCB1Bandit

__all__ = [
    "BaseBandit",
    "EpsilonGreedyBandit",
    "PACSuccessiveEliminationBandit",
    "SoftmaxBandit",
    "UCB1Bandit",
]
