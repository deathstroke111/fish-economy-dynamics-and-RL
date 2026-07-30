"""Small online neural feature encoder for contextual bandits."""

import math
import random
from typing import List, Sequence, Tuple

Vector = List[float]


class NeuralStateEncoder:
    """Two-layer tanh encoder trained online from bandit rewards.

    The encoder maps a compact state vector into a denser latent embedding that the
    contextual bandits can use as their feature space.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 8,
        embedding_dim: int = 6,
        learning_rate: float = 0.003,
        seed: int = 0,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0 or embedding_dim <= 0:
            raise ValueError("hidden_dim and embedding_dim must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.random = random.Random(seed)

        self.w1 = self._init_matrix(hidden_dim, input_dim)
        self.b1 = [0.0 for _ in range(hidden_dim)]
        self.w2 = self._init_matrix(embedding_dim, hidden_dim)
        self.b2 = [0.0 for _ in range(embedding_dim)]

    def encode(self, context: Sequence[float]) -> Vector:
        """Return the latent embedding for a raw context."""

        _, _, _, embedding = self._forward(context)
        return embedding

    def update(self, context: Sequence[float], theta: Sequence[float], reward: float) -> None:
        """Move the encoder so the selected arm's linear head predicts reward better."""

        _, hidden, _, embedding = self._forward(context)
        prediction = sum(weight * value for weight, value in zip(theta, embedding))
        error = prediction - reward
        error = max(min(error, 500.0), -500.0)

        grad_embedding = [error * weight for weight in theta]
        grad_pre_embedding = [
            grad * self._tanh_prime(value) for grad, value in zip(grad_embedding, embedding)
        ]

        grad_w2 = [
            [grad_value * hidden_value for hidden_value in hidden]
            for grad_value in grad_pre_embedding
        ]
        grad_b2 = list(grad_pre_embedding)

        grad_hidden = self._mat_vec_mul(self._transpose(self.w2), grad_pre_embedding)
        grad_pre_hidden = [
            grad * self._tanh_prime(value) for grad, value in zip(grad_hidden, hidden)
        ]

        grad_w1 = [
            [grad_value * input_value for input_value in context]
            for grad_value in grad_pre_hidden
        ]
        grad_b1 = list(grad_pre_hidden)

        self._apply_gradients(grad_w1, grad_b1, grad_w2, grad_b2)

    def _forward(self, context: Sequence[float]) -> Tuple[Vector, Vector, Vector, Vector]:
        x = list(context)
        if len(x) != self.input_dim:
            raise ValueError(f"Expected context of length {self.input_dim}, got {len(x)}")

        hidden_pre = self._add_vector(self._mat_vec_mul(self.w1, x), self.b1)
        hidden = [math.tanh(value) for value in hidden_pre]
        embedding_pre = self._add_vector(self._mat_vec_mul(self.w2, hidden), self.b2)
        embedding = [math.tanh(value) for value in embedding_pre]
        return hidden_pre, hidden, embedding_pre, embedding

    def _apply_gradients(
        self,
        grad_w1: Sequence[Sequence[float]],
        grad_b1: Sequence[float],
        grad_w2: Sequence[Sequence[float]],
        grad_b2: Sequence[float],
    ) -> None:
        for row_index, row in enumerate(grad_w1):
            for col_index, value in enumerate(row):
                self.w1[row_index][col_index] -= self.learning_rate * self._clip(value)
        for index, value in enumerate(grad_b1):
            self.b1[index] -= self.learning_rate * self._clip(value)
        for row_index, row in enumerate(grad_w2):
            for col_index, value in enumerate(row):
                self.w2[row_index][col_index] -= self.learning_rate * self._clip(value)
        for index, value in enumerate(grad_b2):
            self.b2[index] -= self.learning_rate * self._clip(value)

    def _init_matrix(self, rows: int, cols: int) -> List[List[float]]:
        scale = 1.0 / math.sqrt(max(cols, 1))
        return [
            [self.random.uniform(-scale, scale) for _ in range(cols)]
            for _ in range(rows)
        ]

    @staticmethod
    def _add_vector(left: Sequence[float], right: Sequence[float]) -> Vector:
        return [lhs + rhs for lhs, rhs in zip(left, right)]

    @staticmethod
    def _mat_vec_mul(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
        return [sum(value * weight for value, weight in zip(row, vector)) for row in matrix]

    @staticmethod
    def _transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
        return [list(column) for column in zip(*matrix)]

    @staticmethod
    def _tanh_prime(value: float) -> float:
        return 1.0 - value * value

    @staticmethod
    def _clip(value: float, limit: float = 5.0) -> float:
        return max(min(value, limit), -limit)
