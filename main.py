import argparse
from sys import exit
from typing import Callable, Dict

import pygame

from engine import Game, Direction, Snake, Map


COLORS: Dict[str, pygame.Color] = {
    'black': pygame.Color(0, 0, 0),
    'white': pygame.Color(255, 255, 255),
    'red': pygame.Color(255, 0, 0),
    'green': pygame.Color(0, 255, 0),
    'blue': pygame.Color(0, 0, 255),
}


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


def session(ai_func: Callable[[Game], Direction]):
    difficulty = 30
    map_size = 72, 48

    game = Game(
        Map(map_size),
        difficulty,
        Snake()
    )
    game.init()

    change_to = game.snake.direction

    while True:
        print(f'Position: {game.snake.position}')
        change_to = ai_func(game)

        if not Direction.is_opposite(change_to, game.snake.direction):
            game.snake.direction = change_to

        game.step()

        if game.is_game_over:
            print('Game is over!')
            break


def ui_session(ai_func: Callable[[Game], Direction], by_keyboard: bool = False):
    check_errors = pygame.init()
    if check_errors[1] > 0:
        print(f'Had {check_errors[1]} errors when initialising game, exiting...')
        exit(-1)
    else:
        print('Game successfully initialised')

    difficulty = 30
    frame_size_x = 720
    frame_size_y = 480

    pygame.display.set_caption('Snake AI')
    game_window = pygame.display.set_mode((frame_size_x, frame_size_y))
    fps_controller = pygame.time.Clock()

    game = Game(
        Map(size=(frame_size_x // 10, frame_size_y // 10)),
        difficulty,
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
        '-n', '--no-window',
        action='store_true',
        help='(flag) Do not display game on the screen'
    )
    parser.add_argument(
        '-k', '--keyboard',
        action='store_true',
        help='(flag) Control snake by keyboard'
    )

    args = parser.parse_args()
    
    if bool(args.no_window):
        session(test_ai)
    else:
        ui_session(test_ai, bool(args.keyboard))
