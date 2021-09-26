from tensorflow import keras
import numpy as np

from engine import Game, Direction
from nn1 import change_direction, generate_observation


prev_observation = np.array([0, 0, 0])
model = keras.models.load_model('./models/snake_model1')


def snake_ai(game: Game) -> Direction:
    global prev_observation, model
    actions = [-1, 0, 1]

    features = np.array([
        np.append(prev_observation, -1),
        np.append(prev_observation, 0),
        np.append(prev_observation, 1)
    ])
    predictions = model.predict(features)
    action = actions[np.argmax(predictions)]

    prev_observation = generate_observation(game)

    return change_direction(key=action, snake_direction=game.snake.direction)
