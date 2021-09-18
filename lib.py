"""Module with helper functions, such as some metrics
"""
from typing import Tuple
from math import sqrt


def manhetten_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def euclidean_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
