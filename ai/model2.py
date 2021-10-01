from keras.engine.sequential import Sequential
from tensorflow import keras
import numpy as np

from engine import Game, Direction
from nn2 import where_snake_blocked, get_angle, change_direction


model: Sequential = keras.models.load_model('./models/snake_model2')


def snake_ai(game: Game) -> Direction:
    global model
    actions = [-1, 0, 1]

    snake_pos, food_pos, _, _ = game.get_info()

    prev_observation = where_snake_blocked(game)
    angle = get_angle(snake_pos[0], food_pos)

    features = np.array([
        np.append(prev_observation, [angle, -1]),
        np.append(prev_observation, [angle, 0]),
        np.append(prev_observation, [angle, 1])
    ])

    predictions = model.predict(features)
    predictions = [np.argmax(i) for i in predictions]

    action = actions[np.argmax(predictions)]

    return change_direction(key=action, snake_direction=game.snake.direction)
