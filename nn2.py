from math import atan2, pi
from random import randint
from statistics import mean
from typing import List, Tuple

import numpy as np
from tflearn import DNN
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

from engine import Snake, Game, Direction, Map


def add_action(observation: np.ndarray, action: int) -> np.ndarray:
    """Append action feature to observation vector in right order

    Args:
        observation (np.ndarray): Snake's observation vector
        action (int): Action to do

    Returns:
        np.ndarray: Vector with action and observation
    """
    return np.append([action], observation)


def is_direction_blocked(snake: List[Tuple[int, int]], dir: np.ndarray) -> bool:
    point = np.array(snake[0]) + np.array(dir)
    return point.tolist() in snake[:-1] or\
        point[0] == 0 or point[1] == 0 or point[0] == 19 or point[1] == 19


def snake_direction_vector(snake_body: List[Tuple[int, int]]) -> np.ndarray:
    return np.array(snake_body[0]) - np.array(snake_body[1])


def food_direction_vector(snake_pos: tuple, food_pos: tuple) -> np.ndarray:
    return np.array(food_pos) - np.array(snake_pos)


def turn_to_left(vector: np.ndarray) -> np.ndarray:
    return np.array([-vector[1], vector[0]])


def turn_to_right(vector: np.ndarray) -> np.ndarray:
    return np.array([vector[1], -vector[0]])


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def get_angle(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_vector(a)
    b = normalize_vector(b)
    return atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / pi


def get_game_action(snake_body: List[Tuple[int, int]], action: int) -> list:
    """Get new Direction depending on the action and present Direction

    Returns:
        Direction: new direction
    """
    # directions and corresponding changes in the coordinate system
    vectors_and_keys = [
        [[-1, 0], Direction.LEFT],
        [[0, 1], Direction.DOWN],
        [[1, 0], Direction.RIGHT],
        [[0, -1], Direction.UP]
    ]
    snake_direction = snake_direction_vector(snake_body) 
    new_direction = snake_direction
    if action == -1:
        new_direction = turn_to_left(snake_direction)
    elif action == 1:
        new_direction = turn_to_right(snake_direction)

    for pair in vectors_and_keys:
        if pair[0] == new_direction.tolist():
            game_action = pair[1]

    return game_action


def generate_action(snake_body: List[Tuple[int, int]]) -> Tuple[int, Direction]:
    """Randomize action

    Returns:
        int: Direction (the action and new Direction)
    """
    action = randint(0, 2) - 1
    return action, get_game_action(snake_body, action)


def generate_observation(game: Game) -> np.ndarray:
    """Check is snake blocked on all directions
    and get angle between snake and food

    Returns:
        np.ndarray([int, int, int, float]): blocking flags and angle
    """
    snake = game.snake.body
    snake_direction = snake_direction_vector(snake)
    food_direction = food_direction_vector(
        game.snake.position, game.food_pos
    )

    barrier_left = is_direction_blocked(snake, turn_to_left(snake_direction))
    barrier_front = is_direction_blocked(snake, snake_direction)
    barrier_right = is_direction_blocked(snake, turn_to_right(snake_direction))
    angle = get_angle(snake_direction, food_direction)

    return np.array([
        int(barrier_left), int(barrier_front), int(barrier_right), angle
    ])


class SnakeNN:
    def __init__(self,
            initial_games: int = 2000,
            test_games: int = 100,
            goal_steps: int = 2000,
            lr: float = 1e-2,
            filename: str = './models/fanaev.tflearn'
        ):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.game = Game(Map([20, 20]), Snake())

    def get_training_data(self) -> list:
        """Generate training data 
        
        Returns: [
            [(x_11, x_12, x_13, x_14, x_15), y_1],
            ...
            [(x_i1, x_i2, x_i3, x_i4, x_i5), y_i]
        ], where
        
        x_i1, x_i2, x_i3: {0,1} (is snake blocked on the left, front, right), 
        x_i4: float (angle between snake's head and food),
        x_i5: {-1, 0, 1} (snake moved left,front or right)
        y_i: {-1,0,1} (is game done and is distance lower)
        """
        training_data = []
        for _ in range(self.initial_games):
            self.game.init()
            pre_score = self.game.score
            # get features (x1, x2, x3, x4)
            prev_observation = generate_observation(self.game)
            prev_food_distance = self.get_food_distance()

            for _ in range(self.goal_steps):
                action, self.game.snake.direction = generate_action(self.game.snake.body)
                self.game.step()
                _, _, done, score = self.game.get_info()
                if done:
                    # if game is done answer will be -1
                    training_data.append(
                        [add_action(prev_observation, action), -1]
                    )
                    break
                else:
                    answer = 0
                    food_distance = self.get_food_distance()
                    if score > pre_score or food_distance < prev_food_distance:
                        answer = 1

                    training_data.append(
                        [add_action(prev_observation, action), answer]
                    )
                    prev_observation = generate_observation(self.game)
                    prev_food_distance = food_distance

        return training_data

    def get_food_distance(self) -> float:
        """Calculate distance between snake and food

        Returns:
            float: Distance
        """
        return np.linalg.norm(food_direction_vector(
            self.game.snake.position,
            self.game.food_pos
        ))

    def model(self) -> DNN:
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 35, activation='relu')
        network = fully_connected(network, 25, activation='relu')
        network = fully_connected(network, 15, activation='relu')
        network = fully_connected(network, 5, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        network = regression(
            network,
            optimizer='adam',
            learning_rate=self.lr,
            loss='mean_square',
            name='target'
        )
        model = DNN(network, tensorboard_dir='log')
        return model

    def train_model(self, training_data: list, model: DNN):
        X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(X, y, n_epoch=5, shuffle=True, run_id=self.filename)
        model.save(self.filename)
        return model

    def test_model(self):
        """Calculate average steps and score of fitted snake
        """
        model = self.train_model(
            training_data=self.get_training_data(), model=self.model()
        )
        steps_arr = []
        scores_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            self.game.init()
            score = self.game.score
            prev_observation = generate_observation(self.game)
            for _ in range(self.goal_steps):
                predictions = []
                # predict for different actions
                for action in range(-1, 2):
                    input_ = add_action(prev_observation, action)
                    input_ = input_.reshape(-1, 5, 1)
                    predictions.append(model.predict(input_))
                action = np.argmax(np.array(predictions))
                self.game.snake.direction = get_game_action(
                    self.game.snake.body,
                    action - 1
                )
                self.game.step()
                _, _, done, score = self.game.get_info()
                game_memory.append([prev_observation, action])
                if done:
                    print(prev_observation)
                    break
                else:
                    prev_observation = generate_observation(self.game)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        print('Average steps:', mean(steps_arr))
        print('Average score:', mean(scores_arr))


if __name__ == "__main__":
    SnakeNN().test_model()
