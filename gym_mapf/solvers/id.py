"""Independence Detection Algorithm"""

from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.solvers.utils import (detect_conflict, best_joint_policy)
from gym_mapf.solvers.value_iteration_agent import plan_with_value_iteration


def group_of_agent(agents_groups, agent_idx):
    groups_of_agent = [i for i in range(len(agents_groups)) if agent_idx in agents_groups[i]]
    # if more than one group contains the given agent something is wrong
    assert len(groups_of_agent) == 1, "agent {} is in more than one group.\n agent groups are:\n {}".format(agent_idx,
                                                                                                            agents_groups)
    return groups_of_agent[0]


def merge_agent_groups(agents_groups, g1, g2):
    return [agents_groups[i] for i in range(len(agents_groups)) if i not in [g1, g2]] + [
        agents_groups[g1] + agents_groups[g2]]


def ID(env: MapfEnv):
    """Solve MAPF gym environment with ID algorithm.

    Return an optimal policy which guarantees no collision is possible.
    """
    agents_groups = [[i] for i in range(env.n_agents)]
    curr_joint_policy = best_joint_policy(env, agents_groups, plan_with_value_iteration)
    conflict = detect_conflict(env, curr_joint_policy)
    while conflict:
        i, _, j, _, _ = conflict
        # merge groups of i and j
        agents_groups = merge_agent_groups(agents_groups,
                                           group_of_agent(agents_groups, i),
                                           group_of_agent(agents_groups, j))

        print("ID merged groups {} and {}, agents groups are {}".format(
            group_of_agent(agents_groups, i),
            group_of_agent(agents_groups, j),
            agents_groups))

        # solve again with the new agent groups
        curr_joint_policy = best_joint_policy(env, agents_groups, plan_with_value_iteration)

        # find a new conflict
        conflict = detect_conflict(env, curr_joint_policy)

    return curr_joint_policy
