from typing import Dict, Callable
from sys import exit

import pygame

from engine import Game, Direction, Snake, Map


_COLORS: Dict[str, pygame.Color] = {
    'black': pygame.Color(0, 0, 0),
    'grey': pygame.Color(100, 100, 100),
    'white': pygame.Color(255, 255, 255),
    'red': pygame.Color(220, 0, 0),
    'green': pygame.Color(0, 230, 0),
    'dark_green': pygame.Color(0, 100, 0),
    'blue': pygame.Color(0, 0, 255),
}


def _user_input(events) -> Direction:
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


def _draw_map(game: Game, window):
    window.fill(_COLORS['white'])

    H = 10

    size = game.map.size
    for i in range(0, size[0]):
        pygame.draw.line(
            window,
            _COLORS['grey'],
            [i * H, 0], [i * H, size[1] * H],
            width=3 if i in [0, size[0]] else 1
        )
    for i in range(0, size[1]):
        pygame.draw.line(
            window,
            _COLORS['grey'],
            [0, i * H], [size[0] * H, i * H],
            width=3 if i in [0, size[1]] else 1
        )

    pygame.draw.rect(
        window,
        _COLORS['red'],
        pygame.Rect(game.food_pos[0] * H, game.food_pos[1] * H, H, H)
    )
    for pos in game.snake.body:
        pygame.draw.rect(window,
            _COLORS['green'],
            pygame.Rect(pos[0] * H, pos[1] * H, H, H)
        )
    pygame.draw.rect(window,
        _COLORS['dark_green'],
        pygame.Rect(game.snake.position[0] * H, game.snake.position[1] * H, H, H)
    )
    
    for pos in game.map.cells:
        pygame.draw.rect(window,
            _COLORS['black'],
            pygame.Rect(pos[0] * H, pos[1] * H, H, H)
        )

    pygame.display.set_caption(f'Score: {game.score}')
    pygame.display.update()


def session(ai_func: Callable[[Game], Direction], by_keyboard: bool = False):
    check_errors = pygame.init()
    if check_errors[1] > 0:
        print(f'Had {check_errors[1]} errors when initialising game, exiting...')
        exit(-1)
    else:
        print('Game successfully initialised')

    difficulty = 30
    frame_size_x = 720
    frame_size_y = 480
    H = 10

    pygame.display.set_caption('Snake AI')
    game_window = pygame.display.set_mode((frame_size_x, frame_size_y))
    fps_controller = pygame.time.Clock()

    game = Game(
        Map(
            (frame_size_x // H, frame_size_y // H)
        ),
        Snake()
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
            change_to = _user_input(events) or game.snake.direction
        else:
            change_to = ai_func(game)

        if not Direction.is_opposite(change_to, game.snake.direction):
            game.snake.direction = change_to

        game.step()

        _draw_map(game, game_window)
        fps_controller.tick(difficulty)
