"""Indepencde Detection Algorithm"""

from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.solvers.utils import (cross_policies,
                                    detect_conflict,
                                    value_iteration)
from gym_mapf.envs.utils import get_local_view


def best_joint_policy(env, agent_groups):
    local_envs = [get_local_view(env, group) for group in agent_groups]

    policies = []
    for local_env in local_envs:
        r, p = value_iteration(local_env)
        policies.append(p)

    possible_state_counts = [local_env.nS for local_env in local_envs]

    joint_policy = cross_policies(policies, possible_state_counts)

    return joint_policy


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
    curr_joint_policy = best_joint_policy(env, agents_groups)
    env.render_with_policy(0, curr_joint_policy)
    conflict = detect_conflict(env, curr_joint_policy)
    while conflict:
        i, _, j, _, _ = conflict
        # merge groups of i and j
        agents_groups = merge_agent_groups(agents_groups,
                                           group_of_agent(agents_groups, i),
                                           group_of_agent(agents_groups, j))

        # solve again with the new agent groups
        curr_joint_policy = best_joint_policy(env, agents_groups)

        # find a new conflict
        conflict = detect_conflict(env, curr_joint_policy)

    return curr_joint_policy
