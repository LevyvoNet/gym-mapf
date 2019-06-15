import os
from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.utils.grid import MapfGrid

EMPTY_8_8_MAP_FILE = os.path.abspath(os.path.join(__file__, '../../maps/empty-8-8/empty-8-8.map'))
EMPTY_8_8_SCEN_FILE = os.path.abspath(os.path.join(__file__, '../../maps/empty-8-8/empty-8-8-even-1.scen'))
BERLIN_1_256_MAP_FILE = os.path.abspath(os.path.join(__file__, '../../maps/Berlin_1_256/Berlin_1_256.map'))
BERLIN_1_256_SCEN_FILE = os.path.abspath(os.path.join(__file__, '../../maps/Berlin_1_256/Berlin_1_256-even-1.scen'))
MAP_NAME_TO_FILES = {
    'empty-8-8': (EMPTY_8_8_MAP_FILE, EMPTY_8_8_SCEN_FILE),
    'berlin-1-256': (BERLIN_1_256_MAP_FILE, BERLIN_1_256_SCEN_FILE)

}


def parse_scen_file(scen_file, n_agents):
    """Return the agent start locations and the goal locations.

    Args:
        scen_file (str): path to the scenario file.
        n_agents (int): number of agents to read from the scenario (might contain a lot of agents - the more the harder).

    Returns:
        tuple. two lists - one of start locations and one of goal locations (each locations is a tuple of x,y).
    """
    starts = []
    goals = []
    with open(scen_file, 'r') as f:
        lines = iter(f)
        next(lines)
        for i, line in enumerate(lines):
            _, _, _, _, x_start, y_start, x_goal, y_goal, _ = line.split('\t')
            starts.append((int(x_start), int(y_start)))
            goals.append((int(x_goal), int(y_goal)))
            if i == n_agents - 1:
                break

    return starts, goals


def parse_map_file(map_file):
    with open(map_file, 'r') as f:
        lines = f.readlines()

    return lines[4:]


def create_mapf_env(map_name, n_agents):
    map_file, scen_file = MAP_NAME_TO_FILES[map_name]
    grid = MapfGrid(parse_map_file(map_file))
    agents_starts, agents_goals = parse_scen_file(scen_file, n_agents)

    env = MapfEnv(grid, agents_starts, agents_goals)

    return env
