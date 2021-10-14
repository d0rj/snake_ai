import numpy as np

from engine import Game, Direction
from nn2 import SnakeNN, add_action, get_game_action, generate_observation


snake_nn = SnakeNN()
model = snake_nn.model()
model.load(snake_nn.filename)    


def snake_ai(game: Game) -> Direction:
    global model

    prev_observation = generate_observation(game)
    predictions = []
    for action in range(-1, 2):
        input_ = add_action(prev_observation, action)
        input_ = input_.reshape(-1, 5, 1)
        predictions.append(model.predict(input_))

    action = np.argmax(np.array(predictions))
    new_direction = get_game_action(
        game.snake.body,
        action - 1
    )

    return new_direction
