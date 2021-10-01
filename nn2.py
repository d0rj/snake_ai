from operator import floordiv
from numpy.lib.function_base import angle
from tensorflow.python.framework.ops import disable_tensor_equality
from engine import Game, Direction, Map, Snake
import numpy as np
import math
from numpy.random import choice
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
class SnakeNN:
    def __init__(self, initial_games= 1000,test_games = 25, steps=100, map_size = [20,20]):
        self.initial_games = initial_games
        self.steps = steps
        self.test_games = test_games
        self.game = Game(Map(map_size), Snake())
    def get_training_data(self):
        training_data = np.array([])
        for i in range(self.initial_games):
            self.game.init()
            snake_pos = self.game.snake.body
            food_pos = self.game.food_pos
            pre_score = self.game.score
            obstacles = self.where_snake_blocked()
            pre_distance = self.euclidean_distance(snake_pos[0], food_pos)
            angle = self.get_angle(snake_pos[0],food_pos)
            for i in range(self.steps):
                action, self.game.snake.direction = self.generate_action()
                self.game.step()
                done = self.game.is_game_over
                score = self.game.score
                snake_pos = self.game.snake.body
                food_pos = self.game.food_pos
                distance = self.euclidean_distance(snake_pos[0], food_pos)
                if done:
                    training_data = np.append(training_data, np.append(obstacles, [angle,action, -1]))
                    break
                else:
                    if score > pre_score or distance < pre_distance:
                        training_data = np.append(training_data, np.append(obstacles, [angle,action, 1]))
                    else:
                        training_data = np.append(training_data, np.append(obstacles, [angle,action, 0]))
                    pre_score = score
                    pre_distance = distance
                    obstacles = self.where_snake_blocked()
                    angle = self.get_angle(snake_pos[0],food_pos)
        training_data = training_data.reshape(int(len(training_data)/6),6)
        return training_data

    def where_snake_blocked(self):
            snake_direction = self.game.snake.direction
            pos = self.game.snake.position
            size = self.game.map.size
            body =np.array(self.game.snake.body)
            barriers = np.array([0,0,0])
            if snake_direction == Direction.RIGHT:
                if np.logical_or((pos[0] == size[0] - 1),(True in body[:,0] == pos[0] + 1)).any():
                    barriers[0] = 1
                elif np.logical_or((pos[1] == 0), (True in body[:,1] == pos[1] - 1)).any():
                    barriers[1] = 1
                elif np.logical_or((pos[1] == size[1] - 1) , (True in body[:,1] == pos[1] + 1)).any():
                    barriers[2] = 1
            elif snake_direction == Direction.LEFT:
                if np.logical_or(pos[0] == 0, True in body[:,0] == pos[0] - 1).any():
                    barriers[0] = 1
                elif np.logical_or((pos[1] == size[1] - 1), (True in body[:,1] == pos[1] + 1)).any():
                    barriers[1] = 1
                elif np.logical_or((pos[1] == 0), (True in body[:,1] == pos[1] - 1)).any():
                    barriers[2] = 1
            elif snake_direction == Direction.UP:
                if np.logical_or(pos[1] == size[1] - 1, True in body[:,1] == pos[1] + 1).any():
                    barriers[0] = 1
                elif np.logical_or((pos[0] == size[0] - 1), (True in body[:,0] == pos[0] + 1)).any():
                    barriers[1] = 1
                if np.logical_or((pos[0] == 0), (True in body[:,0] == pos[0] - 1)).any():
                    barriers[2] = 1
            elif snake_direction == Direction.DOWN:
                if np.logical_or((pos[1] == 0), (True in body[:,1] == pos[1] - 1)).any():
                    barriers[0] = 1
                elif np.logical_or((pos[0] == 0), (True in body[:,0] == pos[0] - 1)).any():
                    barriers[1] = 1
                elif np.logical_or((pos[0] == size[0] - 1), (True in body[:,0] == pos[0] + 1)).any():
                    barriers[2] = 1
            
            return barriers
    def generate_action(self):
            action = choice([-1,0,1])       
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

    def change_direction(self,key):
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
    def euclidean_distance(self, x, y):
        return np.linalg.norm(np.array(x)-np.array(y))
    def get_angle(self, x, y):
        x = np.array(x)
        y = np.array(y)
        if np.linalg.norm(x) != 0:
            x = x / np.linalg.norm(x)
        y = y/np.linalg.norm(y)
        return math.atan2(x[0] * y[1] - x[1] * y[0], x[0] * y[0] + x[1] * y[1]) / math.pi 
    def train_model(self):
        training_data = self.get_training_data()
        X = training_data[:,[0,1,2,3,4]]
        y = to_categorical(training_data[:,5], num_classes = 3)
        model = keras.Sequential()
        model.add(Dense(25, activation='relu', input_dim=5))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
        model.fit(X, y, batch_size=32, epochs=10)
        model.save('model')
        return model
    
    def test_model(self):
        model = self.train_model()
        steps_arr = []
        scores_arr = []
        actions = [-1,0,1]
        for _ in range(self.test_games):
            steps = 0           
            self.game.init()
            snake_pos, food_pos, _, score = self.game.get_info()
            angle = self.get_angle(snake_pos[0],food_pos)
            prev_observation = self.where_snake_blocked()

            for _ in range(self.steps):
                features = np.array([
                    np.append(prev_observation,[angle, -1]),
                    np.append(prev_observation,[angle, 0]),
                    np.append(prev_observation,[angle, 1])
                ])
                predictions = model.predict(features)
                predictions = [np.argmax(i) for i in predictions]
                action = actions[np.argmax(predictions)]
                self.game.snake.direction = self.change_direction(key = action)
                self.game.step()
                snake_pos, food_pos, done, score = self.game.get_info()
                if done:
                    break
                else:
                    prev_observation = self.where_snake_blocked()
                    angle = self.get_angle(snake_pos[0], food_pos)
                    steps += 1
        steps_arr.append(steps)
        scores_arr.append(score)
        return np.mean(steps_arr), np.mean(scores_arr)

if __name__ == '__main__':
    nn = SnakeNN()
    print(nn.test_model())