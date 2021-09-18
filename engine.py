from random import randrange
from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum


class Direction(Enum):
    UP, DOWN, LEFT, RIGHT = 1, -1, 2, -2

    @staticmethod
    def is_opposite(first, second) -> bool:
        return (first.value + second.value) == 0


class Map:
    def __init__(self, size: Tuple[int, int]):
        self.size = size
        self.cells = [[0 for _ in range(size[0])] for _ in range(size[1])]


@dataclass
class Snake:
    direction: Direction
    position: Tuple[int, int] = (10, 5)
    body: List[Tuple[int, int]] =\
        field(default_factory=lambda: [(10, 5), (10 - 1, 5), (10 - 2, 5)])


@dataclass
class Game:
    map: Map
    difficulty: int
    snake: Snake
    score: int = 0
    is_game_over: bool = False

    def init(self):
        self.spawn_food()

    def spawn_food(self):
        self.food_pos = (
            randrange(1, self.map.size[0]),
            randrange(1, self.map.size[1])
        )

    def move_snake(self):
        head_pos = self.snake.position

        if self.snake.direction == Direction.UP:
            self.snake.position = head_pos[0], head_pos[1] - 1
        if self.snake.direction == Direction.DOWN:
            self.snake.position = head_pos[0], head_pos[1] + 1
        if self.snake.direction == Direction.LEFT:
            self.snake.position = head_pos[0] - 1, head_pos[1]
        if self.snake.direction == Direction.RIGHT:
            self.snake.position = head_pos[0] + 1, head_pos[1]

        head_pos = self.snake.position

        self.snake.body.insert(0, head_pos)
        if head_pos[0] == self.food_pos[0] and head_pos[1] == self.food_pos[1]:
            self.score += 1
            self.spawn_food()
        else:
            self.snake.body.pop()

    def is_snake_collides(self) -> bool:
        head_pos = self.snake.position

        if head_pos[0] < 0 or head_pos[0] > self.map.size[0] - 1:
            return True
        if head_pos[1] < 0 or head_pos[1] > self.map.size[1] - 1:
            return True

        for block in self.snake.body[1:]:
            if head_pos[0] == block[0] and head_pos[1] == block[1]:
                return True

        return False

    def step(self):
        if not self.is_game_over:
            self.move_snake()
            self.is_game_over = self.is_snake_collides()
