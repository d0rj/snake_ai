from typing import Callable

from engine import Game, Direction, Snake, Map


def session(ai_func: Callable[[Game], Direction], _: bool = False):
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
