from engine import Game, Direction


count = 0
def snake_ai(game: Game) -> Direction:
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
