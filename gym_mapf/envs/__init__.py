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
        for line in lines:
            _, _, _, _, x_start, y_start, x_goal, y_goal, _ = line.split('\t')
            starts.append((int(x_start), int(y_start)))
            goals.append((int(x_goal), int(y_goal)))

    # TODO: implement efficiently.
    return starts[:n_agents], goals[:n_agents]


# def parse_map_file(map_file):
#     with open(map_file, 'r') as f:
#         lines = iter(f)
#         for _ in range(4):  # skip first 4 lines
#             next(lines)
#
#         while lines:
#             yield lines


def parse_map_file(map_file):
    with open(map_file, 'r') as f:
        lines = f.readlines()

    return lines[4:]
