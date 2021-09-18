import argparse
from sys import exit
from random import randrange
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
from enum import Enum

import pygame


check_errors = pygame.init()
if check_errors[1] > 0:
    print(f'Had {check_errors[1]} errors when initialising game, exiting...')
    exit(-1)
else:
    print('Game successfully initialised')


COLORS: Dict[str, pygame.Color] = {
    'black': pygame.Color(0, 0, 0),
    'white': pygame.Color(255, 255, 255),
    'red': pygame.Color(255, 0, 0),
    'green': pygame.Color(0, 255, 0),
    'blue': pygame.Color(0, 0, 255),
}


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


count = 0
def test_ai(game: Game) -> Direction:
    global count

    count += 1
    if count >= 5:
        count = 0
        if game.snake.direction == Direction.RIGHT:
            return Direction.DOWN
        if game.snake.direction == Direction.DOWN:
            return Direction.LEFT
        if game.snake.direction == Direction.LEFT:
            return Direction.UP
        if game.snake.direction == Direction.UP:
            return Direction.RIGHT

    return game.snake.direction


def user_input(events) -> Direction:
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP or event.key == ord('w'):
                return Direction.UP
            if event.key == pygame.K_DOWN or event.key == ord('s'):
                return Direction.DOWN
            if event.key == pygame.K_LEFT or event.key == ord('a'):
                return Direction.LEFT
            if event.key == pygame.K_RIGHT or event.key == ord('d'):
                return Direction.RIGHT


def draw_map(game: Game, window):
    window.fill(COLORS['black'])
    pygame.draw.rect(
        window,
        COLORS['white'],
        pygame.Rect(game.food_pos[0] * 10, game.food_pos[1] * 10, 10, 10)
    )
    for pos in game.snake.body:
        pygame.draw.rect(window,
            COLORS['green'],
            pygame.Rect(pos[0] * 10, pos[1] * 10, 10, 10)
        )

    pygame.display.update()


def session(ai_func: Callable[[Game], Direction], by_keyboard: bool = False):
    difficulty = 30
    frame_size_x = 720
    frame_size_y = 480

    pygame.display.set_caption('Snake AI')
    game_window = pygame.display.set_mode((frame_size_x, frame_size_y))
    fps_controller = pygame.time.Clock()

    game = Game(
        Map(size=(frame_size_x // 10, frame_size_y // 10)),
        difficulty,
        Snake(
            direction=Direction.RIGHT
        )
    )
    game.init()

    change_to = game.snake.direction

    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))

        if by_keyboard:
            change_to = user_input(events) or game.snake.direction
        else:
            change_to = ai_func(game)

        if not Direction.is_opposite(change_to, game.snake.direction):
            game.snake.direction = change_to

        game.step()

        draw_map(game, game_window)
        fps_controller.tick(difficulty)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run snake session')
    parser.add_argument(
        '-k', '--keyboard',
        action='store_true',
        help='(flag) Control snake by keyboard'
    )

    args = parser.parse_args()
    
    session(test_ai, bool(args.keyboard))
