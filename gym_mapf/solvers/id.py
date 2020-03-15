"""Independence Detection Algorithm"""
import time
from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.solvers.utils import (detect_conflict,
                                    best_joint_policy,
                                    get_local_view)
from gym_mapf.solvers.value_iteration import value_iteration_planning


def group_of_agent(agents_groups, agent_idx):
    groups_of_agent = [i for i in range(len(agents_groups)) if agent_idx in agents_groups[i]]
    # if more than one group contains the given agent something is wrong
    assert len(groups_of_agent) == 1, "agent {} is in more than one group.\n agent groups are:\n {}".format(agent_idx,
                                                                                                            agents_groups)
    return groups_of_agent[0]


def merge_agent_groups(agents_groups, g1, g2):
    return [agents_groups[i] for i in range(len(agents_groups)) if i not in [g1, g2]] + [
        agents_groups[g1] + agents_groups[g2]]


def ID(env: MapfEnv, **kwargs):
    """Solve MAPF gym environment with ID algorithm.

    Args:
        env (MapfEnv): mapf env
        info (dict): information about the run. For ID it will return information about conflicts
            detected during the solving.

    Returns:
          function int->int. The optimal policy, function from state to action.
    """
    info = kwargs.get('info', {})
    start = time.time()  # TODO: use a decorator for updateing info with time measurement
    agents_groups = [[i] for i in range(env.n_agents)]
    info['iterations'] = []
    curr_iter_info = {}
    info['iterations'].append(curr_iter_info)
    curr_iter_info['agent_groups'] = agents_groups
    curr_iter_info['joint_policy'] = {}
    curr_joint_policy = best_joint_policy(env,
                                          agents_groups,
                                          value_iteration_planning,
                                          **{'info': curr_iter_info['joint_policy']})
    conflict = detect_conflict(env, curr_joint_policy, **{'info': curr_iter_info})
    while conflict:
        i, s_i, j, s_j, s_ij = conflict
        local_env_single_agent = get_local_view(env, [i])
        curr_iter_info['conflict'] = (i,
                                      local_env_single_agent.state_to_locations(s_i),
                                      j,
                                      local_env_single_agent.state_to_locations(s_j),
                                      local_env_single_agent.state_to_locations(s_ij))

        # merge groups of i and j
        agents_groups = merge_agent_groups(agents_groups,
                                           group_of_agent(agents_groups, i),
                                           group_of_agent(agents_groups, j))

        print(f'ID merged groups of agent {i} and {j}, groups are {agents_groups}')

        # solve again with the new agent groups
        curr_iter_info = {}  # TODO: maybe a do while to avoid this code duplication?
        info['iterations'].append(curr_iter_info)
        curr_iter_info['agent_groups'] = agents_groups
        curr_iter_info['joint_policy'] = {}
        curr_joint_policy = best_joint_policy(env,
                                              agents_groups,
                                              value_iteration_planning,
                                              **{'info': curr_iter_info['joint_policy']})

        # find a new conflict
        conflict = detect_conflict(env, curr_joint_policy, **{'info': curr_iter_info})

    end = time.time()
    info['ID_time'] = end - start
    return curr_joint_policy
