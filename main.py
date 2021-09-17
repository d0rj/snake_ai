# Forked from https://gist.github.com/rajatdiptabiswas/bd0aaa46e975a4da5d090b801aba0611

import sys
import time
from random import randrange
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum

import pygame


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
    body: List[Tuple[int, int]] = field(default_factory=lambda: [(10, 5), (10 - 1, 5), (10 - 2, 5)])


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



# Difficulty settings
# Easy      ->  10
# Medium    ->  25
# Hard      ->  40
# Harder    ->  60
# Impossible->  120
difficulty = 25

frame_size_x = 720
frame_size_y = 480

check_errors = pygame.init()

if check_errors[1] > 0:
    print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialised')


pygame.display.set_caption('Snake AI')
game_window = pygame.display.set_mode((frame_size_x, frame_size_y))


# Game Over
def game_over():
    my_font = pygame.font.SysFont('times new roman', 90)
    game_over_surface = my_font.render('YOU DIED', True, COLORS['red'])
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (frame_size_x / 2, frame_size_y / 4)
    game_window.fill(COLORS['black'])
    game_window.blit(game_over_surface, game_over_rect)
    pygame.display.flip()
    time.sleep(3)
    pygame.quit()
    sys.exit()


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


def main() -> None:
    fps_controller = pygame.time.Clock()

    snake_pos = [100, 50]
    snake_body = [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]]

    food_pos = [randrange(1, (frame_size_x // 10)) * 10, randrange(1, (frame_size_y // 10)) * 10]
    food_spawn = True

    direction: Direction = Direction.RIGHT
    change_to = direction

    score = 0

    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))

        game = Game(
            Map(size=(frame_size_x // 10, frame_size_y // 10)),
            difficulty,
            Snake(
                direction,
                position=(snake_pos[0] // 10, snake_pos[1] // 10),
                body=[(p[0] // 10, p[1] // 10) for p in snake_body]
            )
        )
        # change_to = user_input(events) or direction
        change_to = test_ai(game)

        # Making sure the snake cannot move in the opposite direction instantaneously
        if not Direction.is_opposite(change_to, direction):
            direction = change_to

        # Moving the snake
        if direction == Direction.UP:
            snake_pos[1] -= 10
        if direction == Direction.DOWN:
            snake_pos[1] += 10
        if direction == Direction.LEFT:
            snake_pos[0] -= 10
        if direction == Direction.RIGHT:
            snake_pos[0] += 10

        # Snake body growing mechanism
        snake_body.insert(0, list(snake_pos))
        if snake_pos[0] == food_pos[0] and snake_pos[1] == food_pos[1]:
            score += 1
            food_spawn = False
        else:
            snake_body.pop()

        # Spawning food on the screen
        if not food_spawn:
            food_pos = [randrange(1, (frame_size_x//10)) * 10, randrange(1, (frame_size_y//10)) * 10]
        food_spawn = True

        # GFX
        game_window.fill(COLORS['black'])
        for pos in snake_body:
            pygame.draw.rect(game_window, COLORS['green'], pygame.Rect(pos[0], pos[1], 10, 10))

        # Snake food
        pygame.draw.rect(game_window, COLORS['white'], pygame.Rect(food_pos[0], food_pos[1], 10, 10))

        # Game Over conditions
        # Getting out of bounds
        if snake_pos[0] < 0 or snake_pos[0] > frame_size_x-10:
            game_over()
        if snake_pos[1] < 0 or snake_pos[1] > frame_size_y-10:
            game_over()
        # Touching the snake body
        for block in snake_body[1:]:
            if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
                game_over()

        pygame.display.update()
        fps_controller.tick(difficulty)


if __name__ == '__main__':
    main()
