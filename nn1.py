# By https://github.com/fanaev

from engine import Direction, Game, Snake, Map
from random import choice
import numpy as np
import tflearn as tf
import keras
from keras.layers import Dense


def change_direction(snake_direction: Direction, key: int):
    if key == 1:
        if snake_direction == Direction.RIGHT:
            return Direction.DOWN
        if snake_direction == Direction.LEFT:
            return Direction.UP
        if snake_direction == Direction.UP:
            return Direction.RIGHT
        if snake_direction == Direction.DOWN:
            return Direction.LEFT
    if key == -1:
        if snake_direction == Direction.RIGHT:
            return Direction.UP
        if snake_direction == Direction.LEFT:
            return Direction.DOWN
        if snake_direction == Direction.UP:
            return Direction.LEFT
        if snake_direction == Direction.DOWN:
            return Direction.RIGHT
    return snake_direction


def generate_observation(game: Game):
    snake_direction = game.snake.direction
    pos = game.snake.position
    size = game.map.size
    barriers = np.array([0, 0, 0])
    if snake_direction == Direction.RIGHT:
        if pos[0] == size[0] - 1:
            barriers[0] = 1
        elif pos[1] == 0:
            barriers[1] = 1
        elif pos[1] == size[1] - 1:
            barriers[2] = 1
    elif snake_direction == Direction.LEFT:
        if pos[0] == 0:
            barriers[0] = 1
        elif pos[1] == size[1] - 1:
            barriers[1] = 1
        elif pos[1] == 0:
            barriers[2] = 1
    elif snake_direction == Direction.UP:
        if pos[1] == size[1] - 1:
            barriers[0] = 1
        elif pos[0] == size[1] - 1:
            barriers[1] = 1
        if pos[0] == 0:
            barriers[2] = 1
    elif snake_direction == Direction.DOWN:
        if pos[1] == 0:
            barriers[0] = 1
        elif pos[0] == 0:
            barriers[1] = 1
        elif pos[0] == size[0] - 1:
            barriers[2] = 1

    return barriers


class SnakeNN:
    def __init__(self,
            initial_games: int = 100,
            test_games: int = 1000,
            goal_steps: int = 100,
            lr: float = 1e-2,
            filename: str = './models/snake_model1'
        ):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.game = Game(Map([20, 20]), Snake())

    def get_training_data(self):
        training_data = np.array([])
        for _ in range(self.initial_games):
            self.game.init()
            prev_observation = generate_observation(self.game)

            for _ in range(self.goal_steps):
                action, self.game.snake.direction = self.generate_action()
                self.game.step()
                done = self.game.is_game_over
                if done:
                    training_data = np.append(
                        training_data,
                        np.append(prev_observation, [action, 0])
                    )
                    break
                else:
                    training_data = np.append(
                        training_data,
                        np.append(prev_observation, [action, 1])
                    )
                    prev_observation = generate_observation(self.game)
        training_data = training_data.reshape(int(len(training_data) / 5), 5)
        return training_data

    def generate_action(self):
        action = choice([-1, 0, 1])       
        snake_direction = self.game.snake.direction
        #1 - right, -1 - left, 0 - forward
        if snake_direction == Direction.RIGHT:
            if action == 1:
                snake_direction = Direction.DOWN
            elif action == -1:
                snake_direction = Direction.UP
        elif snake_direction == Direction.LEFT:
            if action == 1:
                snake_direction = Direction.UP
            elif action == -1:
                snake_direction = Direction.DOWN
        elif snake_direction == Direction.UP:
            if action == 1:
                snake_direction = Direction.RIGHT
            elif action == -1:
                snake_direction = Direction.LEFT
        elif snake_direction == Direction.DOWN:
            if action == 1:
                snake_direction = Direction.LEFT
            elif action == -1:
                snake_direction = Direction.RIGHT
        return action, snake_direction

    def change_direction(self, key: int):
        snake_direction = self.game.snake.direction
        if key == 1:
            if snake_direction == Direction.RIGHT:
                snake_direction = Direction.DOWN
            elif snake_direction == Direction.LEFT:
                snake_direction = Direction.UP
            elif snake_direction == Direction.UP:
                snake_direction = Direction.RIGHT
            elif snake_direction == Direction.DOWN:
                snake_direction = Direction.LEFT
        elif key == -1:
            if snake_direction == Direction.RIGHT:
                snake_direction = Direction.UP
            elif snake_direction == Direction.LEFT:
                snake_direction = Direction.DOWN
            elif snake_direction == Direction.UP:
                snake_direction = Direction.LEFT
            elif snake_direction == Direction.DOWN:
                snake_direction = Direction.RIGHT
        return snake_direction

    def train_model(self):
        training_data = self.get_training_data()
        X = training_data[:, [0, 1, 2, 3]]
        y = training_data[:, 4]
        model = keras.Sequential()
        model.add(Dense(16, activation='relu', input_dim=4))
        model.add(Dense(8, activation = 'relu'))
        model.add(Dense(4, activation = 'relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
        model.fit(X, y, batch_size=32, epochs=10)
        model.save(self.filename)
        return model

    def test_model(self):
        model = self.train_model()
        steps_arr = []
        actions = [-1, 0, 1]
        for _ in range(self.test_games):
            steps = 0
            
            self.game.init()
            prev_observation = generate_observation(self.game)
            for _ in range(self.goal_steps):
                features = np.array([
                    np.append(prev_observation, -1),
                    np.append(prev_observation, 0),
                    np.append(prev_observation, 1)
                ])
                predictions = model.predict(features)
                action = actions[np.argmax(predictions)]
                self.game.snake.direction = self.change_direction(key=action)
                self.game.step()
                done = self.game.is_game_over
                if done:
                    break
                else:
                    prev_observation = generate_observation(self.game)
                    steps += 1
        steps_arr.append(steps)
        return np.mean(steps_arr)


if __name__ == '__main__':
    nn1 = SnakeNN()
    print(nn1.test_model())
