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
        for line in f:
            _, _, x_start, y_start, x_goal, y_goal, _, _, _ = line.split('\t')
            starts.append((x_start, y_start))
            goals.append((x_goal, y_goal))

    # TODO: implement efficiently.
    return starts[:n_agents], goals[:n_agents]
