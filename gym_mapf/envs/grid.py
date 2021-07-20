import collections
import enum
import random
import gym
from typing import Dict, Callable


class SingleAgentAction(enum.Enum):
    UP = 'UP'
    RIGHT = 'RIGHT'
    DOWN = 'DOWN'
    LEFT = 'LEFT'
    STAY = 'STAY'


ACTIONS = list(SingleAgentAction)

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


class MultiAgentAction:
    def __init__(self, actions: Dict[int, SingleAgentAction]):
        self.agent_to_action = actions

    def __getitem__(self, agent):
        return self.agent_to_action[agent]

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.agent_to_action)})'


class MultiAgentState:
    """A mapping between an agent and its state"""

    def __init__(self, states: Dict[int, SingleAgentState]):
        self.agent_to_state = states

    def __contains__(self, item: SingleAgentState):
        return item in self.agent_to_state.values()

    def __getitem__(self, agent):
        return self.agent_to_state[agent]

    def who_is_here(self, state: SingleAgentState):
        """Return the agents which are in the given state"""
        return [agent for agent in self.agent_to_state
                if self.agent_to_state[agent] == state]

    def __iter__(self):
        return iter(self.agent_to_state.items())

    def agents(self):
        return self.agent_to_state.keys()

    def __eq__(self, other):
        return self.agent_to_state == other.agent_to_state

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.agent_to_state)})'


class MultiAgentStateSpace(gym.Space):
    def __init__(self, n_agents: int, grid):
        super().__init__()
        self.n_agents = n_agents
        self.grid = grid


class MultiAgentActionSpace(gym.Space):
    def __init__(self, n_agents: int):
        super().__init__()
        self.n_agents = n_agents


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
                                for x in range(len(self.map[0]))
                                for y in range(len(self.map[1]))
                                if self.map[x][y] is EmptyCell]

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

    @stay_if_hit_obstacle
    def _up(self, state):
        return SingleAgentState(max(0, state.row-1), state.col)

    @stay_if_hit_obstacle
    def _right(self, state):
        return SingleAgentState(state.row, min(self.max_col, state.col+1))

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
