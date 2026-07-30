"""Small linear algebra helpers for contextual bandits."""

import math
from typing import List


Vector = List[float]
Matrix = List[List[float]]


def identity(size: int, diagonal_value: float = 1.0) -> Matrix:
    return [
        [diagonal_value if row == col else 0.0 for col in range(size)]
        for row in range(size)
    ]


def zeros(size: int) -> Vector:
    return [0.0 for _ in range(size)]


def dot(left: Vector, right: Vector) -> float:
    return sum(lhs * rhs for lhs, rhs in zip(left, right))


def mat_vec_mul(matrix: Matrix, vector: Vector) -> Vector:
    return [dot(row, vector) for row in matrix]


def outer(vector: Vector) -> Matrix:
    return [[lhs * rhs for rhs in vector] for lhs in vector]


def add_matrix_in_place(target: Matrix, delta: Matrix) -> None:
    for row_index, row in enumerate(delta):
        for col_index, value in enumerate(row):
            target[row_index][col_index] += value


def add_scaled_vector_in_place(target: Vector, vector: Vector, scale: float) -> None:
    for index, value in enumerate(vector):
        target[index] += scale * value


def invert_matrix(matrix: Matrix) -> Matrix:
    size = len(matrix)
    augmented = [
        list(row) + [1.0 if row_index == col_index else 0.0 for col_index in range(size)]
        for row_index, row in enumerate(matrix)
    ]

    for pivot_index in range(size):
        pivot_row = max(
            range(pivot_index, size),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            raise ValueError("Matrix is singular and cannot be inverted")
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[pivot_index],
            )

        pivot = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot for value in augmented[pivot_index]]

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            augmented[row_index] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[row_index], augmented[pivot_index])
            ]

    return [row[size:] for row in augmented]


def quadratic_form(matrix: Matrix, vector: Vector) -> float:
    return dot(vector, mat_vec_mul(matrix, vector))


def sqrt_diagonal(diagonal_values: Vector) -> Vector:
    return [math.sqrt(max(value, 0.0)) for value in diagonal_values]
