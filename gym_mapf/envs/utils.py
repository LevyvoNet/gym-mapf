from gym_mapf.envs import map_name_to_files
from gym_mapf.mapf.grid import MapfGrid
from gym_mapf.envs.mapf_env import MapfEnv


def parse_scen_file(scen_file, n_agents):
    """Return the agent start locations and the goal locations.

    Args:
        scen_file (str): path to the scenario file.
        n_agents (int): number of agents to read from the scenario (might contain a lot of agents - the more the harder)

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

    return tuple(starts), tuple(goals)


def parse_map_file(map_file):
    with open(map_file, 'r') as f:
        lines = f.readlines()

    return lines[4:]


def create_mapf_env(map_name, scen_id, n_agents, right_fail, left_fail, reward_of_clash, reward_of_goal,
                    reward_of_living):
    map_file, scen_file = map_name_to_files(map_name, scen_id)
    grid = MapfGrid(parse_map_file(map_file))
    agents_starts, agents_goals = parse_scen_file(scen_file, n_agents)

    env = MapfEnv(grid, agents_starts, agents_goals,
                  right_fail, left_fail, reward_of_clash, reward_of_goal, reward_of_living)

    return env


