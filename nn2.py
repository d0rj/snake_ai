from engine import Snake, Game, Direction, Map
from random import randint
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, mode
from collections import Counter

class SnakeNN:
    def __init__(self, initial_games = 2000, test_games = 100, goal_steps = 2000, lr = 1e-2, filename = 'fanaev.tflearn'):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.game = Game(Map([20,20]), Snake())
        self.vectors_and_keys = [  #directions and corresponding changes in the coordinate system
                [[-1, 0], Direction.LEFT],
                [[0, 1], Direction.DOWN],
                [[1, 0], Direction.RIGHT],
                [[0, -1], Direction.UP]
                ]

    def get_training_data(self):
        '''
        Generate training data 
        
        return: [
            [(x_11, x_12, x_13, x_14, x_15), y_1],
            ...
            [(x_i1, x_i2, x_i3, x_i4, x_i5), y_i]
        ], where x_i1, x_i2, x_i3: {0,1} (is snake blocked on the left, front, right), 
        x_i4: float (angle between snake's head and food),
        x_i5: {-1, 0, 1} (snake moved left,front or right)
        y_i: {-1,0,1} (is game done and is distance lower)
        '''
        training_data = []
        for _ in range(self.initial_games):
            self.game.init()
            pre_score = self.game.score
            prev_observation = self.generate_observation() # get features (x1,x2,x3,x4)
            prev_food_distance = self.get_food_distance() # calculate distance between snake and food
            for _ in range(self.goal_steps):
                action, self.game.snake.direction = self.generate_action() # do random direction where snake is moving
                self.game.step()
                _, _, done, score = self.game.get_info() # check is game over and get score after action
                if done:
                    training_data.append([self.add_action_to_observation(prev_observation, action), -1]) # if game is done answer will be -1
                    break
                else:
                    food_distance = self.get_food_distance() # calculate distance after action
                    if score > pre_score or food_distance < prev_food_distance: # if distance get lower and game isn't done answer will be 1
                        training_data.append([self.add_action_to_observation(prev_observation, action), 1])
                    else:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 0]) #else 0
                    prev_observation = self.generate_observation()
                    prev_food_distance = food_distance
        return training_data

    def generate_action(self):
        '''
        randomize action

        returns: int, Direction (the action and new Direction)
        '''
        action = randint(0,2) - 1
        return action, self.get_game_action(action)

    def get_game_action(self, action):
        '''
        get new Direction depending on the action and present Direction

        returns: Direction
        '''
        snake_direction = self.get_snake_direction_vector() 
        new_direction = snake_direction
        if action == -1:
            new_direction = self.turn_vector_to_the_left(snake_direction)
        elif action == 1:
            new_direction = self.turn_vector_to_the_right(snake_direction)

        for pair in self.vectors_and_keys:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]
        return game_action

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def generate_observation(self):
        '''
        check is snake blocked on all directions and get angle between snake and food

        returns: array([int, int, int, float])
        '''
        snake = self.game.snake.body
        snake_direction = self.get_snake_direction_vector()
        food_direction = self.get_food_direction_vector()
        barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        barrier_front = self.is_direction_blocked(snake, snake_direction)
        barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        angle = self.get_angle(snake_direction, food_direction)
        return np.array([int(barrier_left), int(barrier_front), int(barrier_right), angle])

    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 19 or point[1] == 19    
    def get_snake_direction_vector(self):
        snake = self.game.snake.body
        return np.array(snake[0]) - np.array(snake[1])

    def get_food_direction_vector(self):
        snake = self.game.snake.position
        food = self.game.food_pos
        return np.array(food) - np.array(snake)    
    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def get_food_distance(self):
        return np.linalg.norm(self.get_food_direction_vector())

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def model(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 35, activation='relu')
        network = fully_connected(network, 25, activation='relu')
        network = fully_connected(network, 15, activation='relu')
        network = fully_connected(network, 5, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(X,y, n_epoch = 5, shuffle = True, run_id = self.filename)
        model.save(self.filename)
        return model

    def test_model(self):
        '''
        calculate average steps and score of fitted snake
        '''
        model = self.train_model(training_data=self.get_training_data(), model = self.model())
        steps_arr = []
        scores_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            self.game.init()
            score = self.game.score
            prev_observation = self.generate_observation() # get features
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(-1, 2):
                   predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1))) # predict for different actions
                action = np.argmax(np.array(predictions)) # take the action, which have largest prediction
                self.game.snake.direction = self.get_game_action(action - 1) 
                self.game.step()
                _,_,done,score = self.game.get_info()
                game_memory.append([prev_observation, action])
                if done:
                    print(prev_observation)
                    break
                else:
                    prev_observation = self.generate_observation()
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        print('Average steps:',mean(steps_arr))
        print('Average score:',mean(scores_arr))


if __name__ == "__main__":
    SnakeNN().test_model()