"""Indepencde Detection Algorithm"""

from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.solvers.utils import (cross_policies,
                                    detect_conflict,
                                    value_iteration)
from gym_mapf.envs.utils import get_local_view


def best_joint_policy(env, agent_groups):
    local_envs = [get_local_view(env, group)
                  for group in agent_groups]

    policies = []
    total_reward = 0
    for local_env in local_envs:
        r, p = value_iteration(local_env)
        total_reward += r
        policies.append(p)

    possible_state_counts = [local_env.nS for local_env in local_envs]

    joint_policy = cross_policies(policies, possible_state_counts)

    return joint_policy


def ID(env: MapfEnv):
    """Solve MAPF gym environment with ID algorithm.

    Return an optimal policy which guarantees no collision is possible.
    """
    agent_to_group = {i: [i] for i in range(env.n_agents)}
    curr_joint_reward, curr_joint_policy = best_joint_policy(env, [[i] for i in range(env.n_agents)])
    conflict = detect_conflict(env, curr_joint_policy)
    while conflict:
        i, _, j, _, _ = conflict
        # merge groups of i and j
        agent_to_group[i] = agent_to_group[j] = agent_to_group[i] + agent_to_group[j]
