import copy
from gym_mapf.utils.state import MapfState

UP = 'UP'
RIGHT = 'RIGHT'
DOWN = 'DOWN'
LEFT = 'LEFT'

ACTIONS = [UP, RIGHT, DOWN, LEFT]


def execute_up(loc, map_height, map_width):
    return max(0, loc[0] - 1), loc[1]


def execute_down(loc, map_height, map_width):
    return min(map_height - 1, loc[0] + 1), loc[1]


def execute_right(loc, map_height, map_width):
    return loc[0], min(map_width - 1, loc[1] + 1)


def execute_left(loc, map_height, map_width):
    return loc[0], max(0, loc[1] - 1)


ACTION_TO_FUNC = {
    UP: execute_up,
    DOWN: execute_down,
    RIGHT: execute_right,
    LEFT: execute_left
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
    new_state = copy.deepcopy(s)
    for i, single_action in enumerate(a):
        exec_func = ACTION_TO_FUNC[single_action]
        map_height, map_width = len(s.map), len(s.map[0])
        new_state.agent_locations[i] = exec_func(new_state.agent_locations[i], map_height, map_width)

    return new_state
