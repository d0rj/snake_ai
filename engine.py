from hashlib import new
from random import randrange
from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum


class Direction(Enum):
    """Moving direction representation

    Values
    ------

    UP, DOWN, LEFT, RIGHT

    Static methods
    --------------

    is_opposite(first, second): bool
        is that two values opposite?
    """
    UP, DOWN, LEFT, RIGHT = 1, -1, 2, -2

    @staticmethod
    def is_opposite(first, second) -> bool:
        """Is two values opposite

        It will be True for arguments: (UP, DOWN), (DOWN, UP), (LEFT, RIGHT) and
        (RIGHT, LEFT)

        Args:
            first (Direction): one object
            second (Direction): another object

        Returns:
            bool: is opposite?
        """
        return (first.value + second.value) == 0


@dataclass
class Map:
    """Representation of map

    Attributes
    ----------

    size: Tuple[int, int]
        constant size of map (width, height) in cells
    cells: List[Tuple[int, int]]
        list of walls coordinates on map
    """
    _size: Tuple[int, int]
    _cells: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def size(self) -> Tuple[int, int]:
        """Map size in cells

        Returns:
            Tuple[int, int]: width, height
        """
        return self._size

    @property
    def cells(self) -> List[Tuple[int, int]]:
        """Positions of walls on map

        Returns:
            List[Tuple[int, int]]: list of (x, y) coordinates of walls
        """
        return self._cells


@dataclass
class Snake:
    """Representation of snake and it's current state

    Attributes
    ----------

    direction: Direction
        snake movement direction
    position: Tuple[int, int]
        position of snake's head (x, y)
    body: List[Tuple[int, int]]
        list of coordinates of snake's body cells
    """
    direction: Direction = Direction.RIGHT
    _position: Tuple[int, int] = (10, 5)
    _body: List[Tuple[int, int]] =\
        field(default_factory=lambda: [(10, 5), (10 - 1, 5), (10 - 2, 5)])

    @property
    def position(self) -> Direction:
        """Position of snake's head

        See:
            `engine.Direction`

        Returns:
            Direction: direction enum
        """
        return self._position

    @property
    def body(self) -> List[Tuple[int, int]]:
        """List of coordinates of snake's body

        Returns:
            List[Tuple[int, int]]: list of coordinates
        """
        return self._body

    def _move_head(self):
        if self.direction == Direction.UP:
            self._position = self._position[0], self._position[1] - 1
        if self.direction == Direction.DOWN:
            self._position = self._position[0], self._position[1] + 1
        if self.direction == Direction.LEFT:
            self._position = self._position[0] - 1, self._position[1]
        if self.direction == Direction.RIGHT:
            self._position = self._position[0] + 1, self._position[1]


@dataclass
class Game:
    """Representation of current game state

    Attributes
    -----------

    map: Map
        current map state
    snake: Snake
        current snake state
    score: int
        count of eaten food by snake in this game
    is_game_over: bool
        is game ended
    food_pos: Tuple[int, int]
        current position of food on map
    """
    _map: Map
    _snake: Snake
    _score: int = 0
    _is_game_over: bool = False
    _food_pos: Tuple[int, int] = (0, 0)

    @property
    def map(self) -> Map:
        """Current game map representation
        
        See:
            :class:`engine.Map`

        Returns:
            Map: map object
        """
        return self._map

    @property
    def snake(self) -> Snake:
        """Current snake state

        See :class:`engine.Snake`

        Returns:
            Snake: snake representation
        """
        return self._snake

    @property
    def score(self) -> int:
        """Current score (count of eaten apples by snake)

        Returns:
            int: score
        """
        return self._score

    @property
    def is_game_over(self) -> bool:
        """True if game is ended (snake eat themselves or bump into wall)

        Returns:
            bool: is game ended?
        """
        return self._is_game_over

    @property
    def food_pos(self) -> Tuple[int, int]:
        """Current position of food on map

        Returns:
            Tuple[int, int]: x and y coordinates
        """
        return self._food_pos

    def init(self):
        self._score = 0
        self._is_game_over = False
        self._spawn_food()

    def _spawn_food(self):
        new_pos = (
            randrange(1, self._map.size[0]),
            randrange(1, self._map.size[1])
        )
        if new_pos in self._snake.body or new_pos in self._map.cells:
            self._spawn_food()
        else:
            self._food_pos = new_pos

    def _move_snake(self):
        self._snake._move_head()
        head_pos = self._snake.position

        self._snake.body.insert(0, head_pos)
        if head_pos[0] == self._food_pos[0] and head_pos[1] == self._food_pos[1]:
            self._score += 1
            self._spawn_food()
        else:
            self._snake.body.pop()

    def is_snake_collides(self) -> bool:
        """Returns `True` if snake bump into walls or themselves

        Returns:
            bool: is snake head collide any object except food
        """
        head_pos = self._snake.position

        if head_pos[0] < 0 or head_pos[0] > self._map.size[0] - 1:
            return True
        if head_pos[1] < 0 or head_pos[1] > self._map.size[1] - 1:
            return True

        for block in self._snake.body[1:]:
            if head_pos[0] == block[0] and head_pos[1] == block[1]:
                return True

        if head_pos in self._map.cells:
            return True

        return False

    def step(self):
        """Create one game step

        If game is ended (`Game.is_game_over` is True) state will not change
        """
        if not self._is_game_over:
            self._move_snake()
            self._is_game_over = self.is_snake_collides()
