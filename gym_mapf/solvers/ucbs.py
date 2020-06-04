import heapq
from math import inf

from gym_mapf.envs import (vector_to_integer)
from gym_mapf.envs.utils import get_local_view
from gym_mapf.envs.mapf_env import (MapfEnv)
from gym_mapf.solvers.utils import CrossedPolicy, detect_conflict
from gym_mapf.solvers.vi import value_iteration_planning


def constraints_to_mask(constraints: list, local_env: MapfEnv):
    """Transform a list of constraints to mask for MapfEnv.

    The mask is a different transition model which will operate on the states in the mask.
    The states in the mask can be, buy does not have to, exist in env.
    """
    ret = {}
    for ctr, (i, s_i, j, s_j, s_forbidden) in enumerate(constraints):
        # states env.nS + ctr+ n represents the n-th constraint in the constrains list.

        new_fake_state = local_env.nS + ctr
        ret[new_fake_state] = {}
        for action in range(local_env.nA):
            reachable_states = [next_s for (prob, next_s, _, _) in local_env.P[new_fake_state][action] if prob != 0]
            if s_forbidden in reachable_states:
                ret[new_fake_state][action] = [(1, local_env.nS, -inf, True)]
            else:
                ret[new_fake_state][action] = local_env.P[s_i][action]

    return ret


def best_joint_policy_under_constraint(env, constraints, low_level_planner):
    """Get a joint policy and its exptected sum of rewards."""
    policies = []
    total_reward = 0
    local_envs = []
    for i in range(env.n_agents):
        local_envs.append(get_local_view(env, [i]))
        # the n-th constraint for agent i takes place in the (env.nS+n+1)-th state of local env of agent i.
        agent_mask = constraints_to_mask(constraints[i], local_envs[i])
        local_envs[i].set_mask(agent_mask)

        policy = low_level_planner(local_envs[i])
        # Assume the low level planner maintains a value table for all states.
        total_reward += policy.v[policy.env.s]
        policies.append(policy)  # solve as if agent i is alone

    possible_states_counts = [local_envs[i].nS for i in range(env.n_agents)]

    joint_policy = CrossedPolicy(env, 1.0, policies)
    # now set the special states on the joint policy
    synced_joint_policy = sync_joint_policy(joint_policy, env, constraints, possible_states_counts)
    # TODO: fix total_reward, it's inaccurate. Calculate it via policy evaluation.
    return total_reward, synced_joint_policy


def get_matching_constraints(vector_joint_state, constraints):
    ret = []
    for i, s_i, a, j, s_j in constraints:
        if vector_joint_state[i] == s_i and vector_joint_state[j] == s_j:
            ret.append((i, s_i, a, j, s_j))

    return ret


def sync_joint_policy(joint_policy, env: MapfEnv, constraints, possible_states_counts):
    """Transform a joint policy from the space including constrains-states to the original one."""

    def joint_policy_synced(s):
        # notice: here there is an assumption that s is a 'normal' state.
        # therefore the possible_states_counts parameter depends only on grid (and not on constraints as well)
        aux_local_env = get_local_view(env, [0])
        n_regular_states = aux_local_env.nS
        locations = env.state_to_locations(s)
        local_states = tuple([aux_local_env.locations_to_state((locations[i],)) for i in range(env.n_agents)])
        for agent_idx in range(env.n_agents):
            # search for a constraint which includes i, s_i
            for ctr, (i, s_i, j, s_j, s_forbidden) in enumerate(constraints[agent_idx]):
                # this state is matching to a constraint.
                # sync the action of the conflicting agents
                # by letting the compromising agent to know he is in a special state.
                # TODO: shouldn't we check that both of the agents might get to s_forbidden?
                if local_states[i] == s_i and local_states[j] == s_j:
                    synced_vector_joint_state = local_states

                    synced_vector_joint_state = synced_vector_joint_state[:i] + (
                        n_regular_states + ctr,) + synced_vector_joint_state[i + 1:]

                    # TODO: support more than one constraint under the same state
                    s = vector_to_integer(synced_vector_joint_state, possible_states_counts, lambda x: x)
                    return joint_policy.act(s)

        return joint_policy.act(s)

    return joint_policy_synced


def UCBS(env):
    # TODO: problematic action can be problematic state to reach instead. find_conflict should solve this.
    curr_constraints = []
    curr_joint_reward, curr_joint_policy = best_joint_policy_under_constraint(env, curr_constraints,
                                                                              value_iteration_planning)
    search_tree = [(curr_joint_reward, curr_constraints, curr_joint_policy)]
    heapq.heapify(search_tree)

    while True:  # TODO: when to stop?
        conflict = detect_conflict(env, curr_joint_policy)
        if not conflict:
            return curr_joint_reward, curr_joint_policy  # This is the best policy which has no conflicts.

        i, s_i, j, s_j, s_forbidden = conflict  # a conflict is where pi_i(s_i) and pi_j(s_j) might cause a collision

        # solve again but now agent i in s_i can't do pi_i[s_i] if j is in state s_j
        heapq.heappush(search_tree,
                       best_joint_policy_under_constraint(env, curr_constraints[:i] +
                                                          (curr_constraints[i] +
                                                           (i, s_i, j, s_j, s_forbidden))
                                                          + curr_constraints[i + 1:],
                                                          value_iteration_planning))

        # now agent j is the one who compromises
        heapq.heappush(search_tree,
                       best_joint_policy_under_constraint(env, curr_constraints[:j] +
                                                          (curr_constraints[j] +
                                                           (j, s_j, i, s_i, s_forbidden))
                                                          + curr_constraints[j + 1:],
                                                          value_iteration_planning))

        curr_joint_reward, curr_constraints, curr_joint_policy = search_tree.pop()
