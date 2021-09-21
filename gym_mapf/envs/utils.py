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


def create_sanity_mapf_env(n_rooms,
                           room_size,
                           n_agents,
                           fail_prob,
                           reward_of_clash,
                           reward_of_goal,
                           reward_of_living,
                           optimization_criteria):
    single_room = ['.' * room_size] * room_size
    grid_lines = single_room[:]
    n_agents_per_room = int(n_agents / n_rooms)
    n_agents_last_room = n_agents - (n_agents_per_room * (n_rooms - 1))
    agents_starts = tuple()
    agents_goals = tuple()

    if n_agents_last_room == 0 or n_agents_per_room == 0:
        raise ValueError(
            f"asked for a sanity env with {n_rooms} rooms  and {n_agents} agents, There are redundant rooms")

    # concatenate n-1 rooms to a single room
    for i in range(n_rooms - 1):
        # Add the extra room to the map
        for line_idx, line in enumerate(grid_lines[:-1]):
            grid_lines[line_idx] = line + '@@' + single_room[line_idx]

        grid_lines[-1] = grid_lines[-1] + '..' + single_room[-1]

    for i in range(n_rooms):
        # Set the new start and goal locations according to current offset
        map_file, scen_file = map_name_to_files(f'empty-{room_size}-{room_size}', (i + 1) % 26)
        if i != n_rooms - 1:
            orig_agents_starts, orig_agents_goals = parse_scen_file(scen_file, n_agents_per_room)
        else:
            orig_agents_starts, orig_agents_goals = parse_scen_file(scen_file, n_agents_last_room)

        new_agents_starts = tuple()
        for start in orig_agents_starts:
            new_start = (start[0], start[1] + (i) * (len(single_room[0]) + 2))
            new_agents_starts += (new_start,)

        new_agents_goals = tuple()
        for goal in orig_agents_goals:
            new_goal = (goal[0], goal[1] + (i) * (len(single_room[0]) + 2))
            new_agents_goals += (new_goal,)

        agents_starts += new_agents_starts
        agents_goals += new_agents_goals

    grid = MapfGrid(grid_lines)

    return MapfEnv(grid,
                   n_agents,
                   agents_starts,
                   agents_goals,
                   fail_prob,
                   reward_of_clash,
                   reward_of_goal,
                   reward_of_living,
                   optimization_criteria)


def create_mapf_env(map_name,
                    scen_id,
                    n_agents,
                    fail_prob,
                    reward_of_clash,
                    reward_of_goal,
                    reward_of_living,
                    optimization_criteria):
    if map_name.startswith('sanity'):
        [n_rooms, room_size] = [int(n) for n in map_name.split('-')[1:]]
        return create_sanity_mapf_env(n_rooms,
                                      room_size,
                                      n_agents,
                                      fail_prob,
                                      reward_of_clash,
                                      reward_of_goal,
                                      reward_of_living,
                                      optimization_criteria)

    map_file, scen_file = map_name_to_files(map_name, scen_id)
    grid = MapfGrid(parse_map_file(map_file))
    agents_starts, agents_goals = parse_scen_file(scen_file, n_agents)
    n_agents = len(agents_goals)

    env = MapfEnv(grid,
                  n_agents,
                  agents_starts,
                  agents_goals,
                  fail_prob,
                  reward_of_clash,
                  reward_of_goal,
                  reward_of_living,
                  optimization_criteria)

    return env


def get_local_view(env: MapfEnv, agent_indexes: list, **kwargs):
    fail_prob = kwargs.get('fail_prob', env.fail_prob)

    vector_local_agents_starts = tuple(itertools.compress(env.agents_starts,
                                                          [1 if x in agent_indexes else 0
                                                           for x in range(env.n_agents)]))

    vector_local_agents_goals = tuple(itertools.compress(env.agents_goals,
                                                         [1 if x in agent_indexes else 0
                                                          for x in range(env.n_agents)]))

    return MapfEnv(env.grid,
                   len(agent_indexes),
                   vector_local_agents_starts,
                   vector_local_agents_goals,
                   fail_prob,
                   env.reward_of_clash,
                   env.reward_of_goal,
                   env.reward_of_living,
                   env.optimization_criteria)


def mapf_env_load_from_json(json_str: str) -> MapfEnv:
    raise NotImplementedError()


def manhattan_distance(env: MapfEnv, s, a1, a2):
    """Return the manhattan distance between the two given agents in the given state"""
    locations = env.state_to_locations(s)
    return abs(locations[a1][0] - locations[a2][0]) + abs(locations[a1][1] - locations[a2][1])
