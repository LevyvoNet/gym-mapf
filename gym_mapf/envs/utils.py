import itertools

from gym_mapf.envs import map_name_to_files
from gym_mapf.envs.grid import MapfGrid
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


def create_sanity_mapf_env(n_agents, right_fail, left_fail, reward_of_clash, reward_of_goal, reward_of_living):
    single_room = [
        '....',
        '....',
        '....',
        '....',
    ]
    grid_lines = single_room[:]
    start = (len(single_room) - 1, 0)
    goal = (0, len(single_room[0]) - 1)

    agents_starts = (start,)
    agents_goals = (goal,)

    # concatenate n-1 rooms to a single room
    for i in range(n_agents - 1):
        for line_idx, line in enumerate(grid_lines[:-1]):
            grid_lines[line_idx] = line + '@@' + single_room[line_idx]

        grid_lines[-1] = grid_lines[-1] + '..' + single_room[-1]

        new_start = (start[0], start[1] + (i + 1) * (len(single_room[0]) + 2))
        new_goal = (goal[0], goal[1] + (i + 1) * (len(single_room[0]) + 2))
        agents_starts += (new_start,)
        agents_goals += (new_goal,)

    grid = MapfGrid(grid_lines)

    return MapfEnv(grid,
                   n_agents,
                   agents_starts,
                   agents_goals,
                   right_fail,
                   left_fail,
                   reward_of_clash,
                   reward_of_goal,
                   reward_of_living)


def create_mapf_env(map_name, scen_id, n_agents, right_fail, left_fail, reward_of_clash, reward_of_goal,
                    reward_of_living):
    if map_name == 'sanity':
        return create_sanity_mapf_env(n_agents, right_fail, left_fail, reward_of_clash, reward_of_goal,
                                      reward_of_living)

    map_file, scen_file = map_name_to_files(map_name, scen_id)
    grid = MapfGrid(parse_map_file(map_file))
    agents_starts, agents_goals = parse_scen_file(scen_file, n_agents)
    n_agents = len(agents_goals)

    env = MapfEnv(grid, n_agents, agents_starts, agents_goals,
                  right_fail, left_fail, reward_of_clash, reward_of_goal, reward_of_living)

    return env


def get_local_view(env: MapfEnv, agent_indexes: list, **kwargs):
    right_fail = kwargs.get('right_fail', env.right_fail)
    left_fail = kwargs.get('left_fail', env.left_fail)

    vector_local_agents_starts = tuple(itertools.compress(env.agents_starts,
                                                          [1 if x in agent_indexes else 0
                                                           for x in range(env.n_agents)]))

    vector_local_agents_goals = tuple(itertools.compress(env.agents_goals,
                                                         [1 if x in agent_indexes else 0
                                                          for x in range(env.n_agents)]))

    return MapfEnv(env.grid, len(agent_indexes), vector_local_agents_starts, vector_local_agents_goals,
                   right_fail, left_fail, env.reward_of_clash, env.reward_of_goal, env.reward_of_living)


def mapf_env_load_from_json(json_str: str) -> MapfEnv:
    raise NotImplementedError()


def manhattan_distance(env: MapfEnv, s, a1, a2):
    """Return the manhattan distance between the two given agents in the given state"""
    locations = env.state_to_locations(s)
    return abs(locations[a1][0] - locations[a2][0]) + abs(locations[a1][1] - locations[a2][1])
