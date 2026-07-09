"""Linear UCB contextual bandit."""

import math
from typing import Dict, List

from .contextual_base import BaseContextualBandit
from .linalg import (
    Matrix,
    Vector,
    add_matrix_in_place,
    add_scaled_vector_in_place,
    identity,
    invert_matrix,
    mat_vec_mul,
    outer,
    quadratic_form,
    zeros,
)


class LinUCBBandit(BaseContextualBandit):
    def __init__(
        self,
        arms,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__(arms=arms, seed=seed)
        dimension = 4
        self.alpha = alpha
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
        best_arm_id = self.arm_ids[0]
        best_score = float("-inf")
        for arm_id in self.arm_ids:
            inverse = invert_matrix(self.A_matrices[arm_id])
            theta = mat_vec_mul(inverse, self.b_vectors[arm_id])
            exploit = sum(feature * weight for feature, weight in zip(context, theta))
            explore = self.alpha * math.sqrt(quadratic_form(inverse, context))
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_arm_id = arm_id
        return best_arm_id

    def update(self, context: List[float], arm_id: int, reward: float) -> None:
        add_matrix_in_place(self.A_matrices[arm_id], outer(context))
        add_scaled_vector_in_place(self.b_vectors[arm_id], context, reward)
        super().update(context, arm_id, reward)
