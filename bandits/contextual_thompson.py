"""Contextual Thompson sampling with diagonal covariance approximation."""

from typing import Dict, List

from .contextual_base import BaseContextualBandit
from .neural_encoder import NeuralStateEncoder
from .linalg import (
    Matrix,
    Vector,
    add_matrix_in_place,
    add_scaled_vector_in_place,
    identity,
    invert_matrix,
    mat_vec_mul,
    outer,
    sqrt_diagonal,
    zeros,
)


class ContextualThompsonBandit(BaseContextualBandit):
    """Approximate linear Thompson sampler using diagonal posterior covariance."""

    def __init__(
        self,
        arms,
        v: float = 0.5,
        lambda_reg: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__(arms=arms, seed=seed)
        self.encoder = NeuralStateEncoder(input_dim=2, seed=seed)
        dimension = self.encoder.embedding_dim
        self.v = v
        self.lambda_reg = lambda_reg
        self.A_matrices: Dict[int, Matrix] = {
            arm_id: identity(dimension, diagonal_value=lambda_reg)
            for arm_id in self.arm_ids
        }
        self.b_vectors: Dict[int, Vector] = {
            arm_id: zeros(dimension)
            for arm_id in self.arm_ids
        }

    def select_action(self, context: List[float]) -> int:
        embedding = self.encoder.encode(context)
        best_arm_id = self.arm_ids[0]
        best_score = float("-inf")
        for arm_id in self.arm_ids:
            inverse = invert_matrix(self.A_matrices[arm_id])
            mean = mat_vec_mul(inverse, self.b_vectors[arm_id])
            stddevs = [self.v * value for value in sqrt_diagonal([inverse[i][i] for i in range(len(inverse))])]
            sampled_theta = [
                self.random.gauss(mu, sigma)
                for mu, sigma in zip(mean, stddevs)
            ]
            score = sum(feature * weight for feature, weight in zip(embedding, sampled_theta))
            if score > best_score:
                best_score = score
                best_arm_id = arm_id
        return best_arm_id

    def update(self, context: List[float], arm_id: int, reward: float) -> None:
        embedding = self.encoder.encode(context)
        add_matrix_in_place(self.A_matrices[arm_id], outer(embedding))
        add_scaled_vector_in_place(self.b_vectors[arm_id], embedding, reward)
        inverse = invert_matrix(self.A_matrices[arm_id])
        theta = mat_vec_mul(inverse, self.b_vectors[arm_id])
        self.encoder.update(context, theta, reward)
        super().update(context, arm_id, reward)
