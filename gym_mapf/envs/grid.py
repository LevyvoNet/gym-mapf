import collections
import enum
import random
import gym
from typing import Callable


class SingleAgentAction(enum.Enum):
    STAY = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


ACTIONS = [a for a in SingleAgentAction]

SingleAgentState = collections.namedtuple('SingleAgentState', ['row', 'col'])


class SingleAgentStateSpace(gym.Space):
    def __init__(self, grid, shape=None, dtype=None):
        super().__init__(None, None)
        self.grid = grid

    def sample(self):
        x = random.randint(0, len(self.grid.map))
        y = random.randint(0, len(self.grid.map[0]))

        return SingleAgentState(x, y)

    def seed(self, seed=None):
        random.seed(self.seed)

    def contains(self, loc: SingleAgentState):
        return loc in self

    def __contains__(self, loc: SingleAgentState):
        return loc.row < len(self.grid.map) and loc.col < len(self.grid.map[0])

    def to_jsonable(self, sample_n):
        raise NotImplementedError()

    def from_jsonable(self, sample_n):
        raise NotImplementedError()


class SingleAgentActionSpace(gym.Space):
    def __init__(self, shape=None, dtype=None):
        super().__init__(None, None)

    def sample(self):
        raise NotImplementedError()

    def seed(self, seed=None):
        random.seed(self.seed)

    def contains(self, x):
        return x in self

    def __contains__(self, x):
        return x in ACTIONS

    def to_jsonable(self, sample_n):
        raise NotImplementedError()

    def from_jsonable(self, sample_n):
        raise NotImplementedError()

    def __iter__(self):
        return


def integer_to_vector(x, options_per_element, n_elements, index_to_element):
    """Return a vector representing an action/state from a given integer.

    Args:
        x (int): the integer to convert.
        n_options_per_element(int): number of options for each element in the vector.
        n_elements (int): the number of elements in the vector to return.
        index_to_element(int=>any): function which converts an integer represents a single option in one of the
            vector elements and return anything that vector contains. For example, a function which returns 'UP' for 0,
            1 for 'RIGHT',etc. Or a function which returns (2,2) given 10 for a 4x4 grid ((2,2) is the 10-th cell of that grid).
    """
    return integer_to_vector_multiple_numbers(x, options_per_element, n_elements, index_to_element)


def vector_to_integer(v, options_per_element, element_to_index):
    return vector_to_integer_multiple_numbers(v, options_per_element, element_to_index)


def integer_to_vector_multiple_numbers(x, n_options_per_element, n_elements, index_to_element):
    """Return a vector representing an action/state from a given integer.

    Args:
        x (int): the integer to convert.
        n_options_per_element(list): number of options for each element in the vector.
        n_elements (int): the number of elements in the vector to return.
        index_to_element(int=>any): function which converts an integer represents a single option in one of the
            vector elements and return anything that vector contains. For example, a function which returns 'UP' for 0,
            1 for 'RIGHT',etc. Or a function which returns (2,2) given 10 for a 4x4 grid ((2,2) is the 10-th cell of that grid).
    """
    ret = tuple()
    for i in range(0, n_elements):
        option_index = x % n_options_per_element[i]
        ret = ret + (index_to_element(option_index),)
        x //= n_options_per_element[i]

    return ret


def vector_to_integer_multiple_numbers(v, n_options_per_element, element_to_index):
    sum = 0
    mul = 1
    for i in range(len(v)):
        if i != 0:
            mul *= n_options_per_element[i - 1]

        sum += element_to_index(v[i]) * mul

    return sum


def vector_action_to_integer(a):
    return vector_to_integer(a, [len(ACTIONS)] * len(a), lambda x: ACTIONS.index(x))


def integer_action_to_vector(a, n_agents):
    return integer_to_vector(a, [len(ACTIONS)] * n_agents, n_agents, lambda n: ACTIONS[n])


class ObstacleCell:
    pass


class EmptyCell:
    pass


CELL_TO_CHAR = {
    EmptyCell: '.',
    ObstacleCell: '@'
}

CHAR_TO_CELL = {
    '.': EmptyCell,
    '@': ObstacleCell

}


# object here is actually MapfGrid, this is a forward declaration.
def stay_if_hit_obstacle(exec_func: Callable[[object, SingleAgentState], SingleAgentState]):
    def new_exec_func(self: object, state: SingleAgentState):
        next_state = exec_func(self, state)
        if self[next_state] is ObstacleCell:
            return state
        else:
            return next_state

    return new_exec_func


class MapfGrid:
    def __init__(self, map_lines):
        self.map = []
        for line in map_lines:
            line = line.strip()
            new_line = [CHAR_TO_CELL[char] for char in line]
            self.map.append(new_line)

        self.max_row = len(self.map) - 1
        self.max_col = len(self.map[0]) - 1
        self.valid_locations = [(x, y)
                                for x in range(len(self.map))
                                for y in range(len(self.map[0]))
                                if self.map[x][y] is EmptyCell]
        self.loc_to_int = {loc: i for i, loc in enumerate(self.valid_locations)}

        self.action_to_func = {
            SingleAgentAction.UP: self._up,
            SingleAgentAction.RIGHT: self._right,
            SingleAgentAction.DOWN: self._down,
            SingleAgentAction.LEFT: self._left,
            SingleAgentAction.STAY: self._stay,
        }

    def __getitem__(self, state: SingleAgentState):
        return self.map[state.row][state.col]

    def __iter__(self):
        for col_idx in range(len(self.map[0])):
            for line_idx in range(len(self.map)):
                yield (line_idx, col_idx)

    def __len__(self):
        return len(self.map)

    def __eq__(self, other):
        return self.map == other.map

    def locations_to_int(self, locs):
        local_state_vector = tuple([self.loc_to_int[loc] for loc in locs])
        return vector_to_integer(local_state_vector, [len(self.valid_locations)] * len(local_state_vector), lambda x: x)

    @stay_if_hit_obstacle
    def _up(self, state):
        return SingleAgentState(max(0, state.row - 1), state.col)

    @stay_if_hit_obstacle
    def _right(self, state):
        return SingleAgentState(state.row, min(self.max_col, state.col + 1))

    @stay_if_hit_obstacle
    def _down(self, state):
        return SingleAgentState(min(self.max_row, state.row + 1), state.col)

    @stay_if_hit_obstacle
    def _left(self, state):
        return SingleAgentState(state.row, max(0, state.col - 1))

    @stay_if_hit_obstacle
    def _stay(self, state):
        return SingleAgentState(state.row, state.col)

    def next_state(self, state: SingleAgentState, action: SingleAgentAction):
        return self.action_to_func[action](state)
