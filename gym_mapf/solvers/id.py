"""Independence Detection Algorithm"""
import time
from typing import Callable

from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.solvers.utils import (detect_conflict,
                                    solve_independently_and_cross,
                                    get_local_view,
                                    Policy,
                                    Planner)


def group_of_agent(agents_groups, agent_idx):
    groups_of_agent = [i for i in range(len(agents_groups)) if agent_idx in agents_groups[i]]
    # if more than one group contains the given agent something is wrong
    assert len(groups_of_agent) == 1, "agent {} is in more than one group.\n agent groups are:\n {}".format(agent_idx,
                                                                                                            agents_groups)
    return groups_of_agent[0]


def merge_agent_groups(agents_groups, g1, g2):
    return [agents_groups[i] for i in range(len(agents_groups)) if i not in [g1, g2]] + [
        agents_groups[g1] + agents_groups[g2]]




class IdPlanner(Planner):
    def __init__(self, low_level_planner: Planner):
        super().__init__()
        self.low_level_planner = low_level_planner

    def plan(self, env: MapfEnv, **kwargs) -> Policy:
        """Solve MAPF gym environment with ID algorithm.

        Args:
            env (MapfEnv): mapf env
            info (dict): information about the run. For ID it will return information about conflicts
                detected during the solving.
            low_level_planner (Callable[[MapfEnv], Policy]): a planner function which receives env and returns a policy.

        Returns:
              function int->int. The optimal policy, function from state to action.
        """
        start = time.time()  # TODO: use a decorator for updating info with time measurement
        agents_groups = [[i] for i in range(env.n_agents)]
        self.info['iterations'] = []
        curr_iter_info = {}
        self.info['iterations'].append(curr_iter_info)
        curr_iter_info['agent_groups'] = agents_groups
        curr_iter_info['joint_policy'] = {}
        curr_joint_policy = solve_independently_and_cross(env,
                                                          agents_groups,
                                                          self.low_level_planner,
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
            self.info['iterations'].append(curr_iter_info)
            curr_iter_info['agent_groups'] = agents_groups
            curr_iter_info['joint_policy'] = {}
            curr_joint_policy = solve_independently_and_cross(env,
                                                              agents_groups,
                                                              self.low_level_planner,
                                                              **{'info': curr_iter_info['joint_policy']})

            # find a new conflict
            conflict = detect_conflict(env, curr_joint_policy, **{'info': curr_iter_info})

        end = time.time()
        self.info['ID_time'] = end - start
        return curr_joint_policy

    def dump_to_str(self):
        pass

    @staticmethod
    def load_from_str(json_str: str) -> object:
        pass
