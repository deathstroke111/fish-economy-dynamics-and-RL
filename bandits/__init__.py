"""Bandit algorithms used in FishEconomy experiments."""

from .base import BaseBandit
from .contextual_base import BaseContextualBandit
from .contextual_thompson import ContextualThompsonBandit
from .discretized_contextual import DiscretizedContextualBandit
from .epsilon_greedy import EpsilonGreedyBandit
from .linucb import LinUCBBandit
from .pac import PACSuccessiveEliminationBandit
from .softmax import SoftmaxBandit
from .ucb import UCB1Bandit

__all__ = [
    "BaseBandit",
    "BaseContextualBandit",
    "ContextualThompsonBandit",
    "DiscretizedContextualBandit",
    "EpsilonGreedyBandit",
    "LinUCBBandit",
    "PACSuccessiveEliminationBandit",
    "SoftmaxBandit",
    "UCB1Bandit",
]
