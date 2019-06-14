import copy
from gym_mapf.utils.state import MapfState
from gym_mapf.utils.grid import ObstacleCell

UP = 'UP'
RIGHT = 'RIGHT'
DOWN = 'DOWN'
LEFT = 'LEFT'
STAY = 'STAY'

ACTIONS = [UP, RIGHT, DOWN, LEFT, STAY]


def stay_if_hit_obstacle(exec_func):
    def new_exec_func(loc, map):
        new_loc = exec_func(loc, map)
        if map[new_loc] is ObstacleCell:
            return loc

        return new_loc

    return new_exec_func


@stay_if_hit_obstacle
def execute_up(loc, map):
    return max(0, loc[0] - 1), loc[1]


@stay_if_hit_obstacle
def execute_down(loc, map):
    return min(len(map) - 1, loc[0] + 1), loc[1]


@stay_if_hit_obstacle
def execute_right(loc, map):
    return loc[0], min(len(map[0]) - 1, loc[1] + 1)


@stay_if_hit_obstacle
def execute_left(loc, map):
    return loc[0], max(0, loc[1] - 1)


def execute_stay(loc, _):
    return loc


ACTION_TO_FUNC = {
    UP: execute_up,
    DOWN: execute_down,
    RIGHT: execute_right,
    LEFT: execute_left,
    STAY: execute_stay
}


def execute_action(s, a):
    """Execute joint action a on state s.

    Args:
        s (MapfState): representation of the map and the agents locations
        a (tuple): joint action such as (UP, UP, RIGHT, DOWN)

    Return:
        MapfState. representation of the new state
    """
    # TODO: support stay action
    # TODO: if collide with an obstacle should remain in place.
    new_state = copy.deepcopy(s)
    for i, single_action in enumerate(a):
        exec_func = ACTION_TO_FUNC[single_action]
        new_state.agent_locations[i] = exec_func(new_state.agent_locations[i], s.map)

    return new_state
