import heapq
from collections import Counter
from typing import Callable

from gym_mapf.envs import (integer_to_vector, vector_to_integer)
from gym_mapf.envs.utils import get_local_view
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_state_to_integer,
                                    vector_action_to_integer,
                                    integer_state_to_vector,
                                    integer_action_to_vector)
from gym_mapf.solvers.value_iteration_agent import ValueIterationAgent


def value_iteration(env):
    """Get optimal policy derived from value iteration and its expected reward"""
    vi_agent = ValueIterationAgent()
    vi_agent.train(env, max_time=10)

    def policy_int_output(s):
        return int(vi_agent.policy[s])

    return vi_agent.optimal_v[env.s], policy_int_output


def constraints_to_mask(constraints: list, local_env: MapfEnv):
    """Transform a list of constraints to mask for MapfEnv.

    The mask is a different transition model which will operate on the states in the mask.
    The states in the mask can be, buy does not have to, exist in env.
    """
    """Return a mask represents the given constrains for agent i."""
    ret = {}
    for ctr, (i, s_i, j, s_j, s_forbidden) in enumerate(constraints):
        # states env.nS + ctr+ n represents the n-th constraint in the constrains list.

        new_fake_state = local_env.nS + ctr
        ret[new_fake_state] = {}
        for action in range(local_env.nA):
            reachable_states = [next_s for (prob, next_s, _, _) in local_env.P[new_fake_state][action] if prob != 0]
            if s_forbidden in reachable_states:
                ret[new_fake_state][action] = [(1, local_env.nS, -100, True)]
            else:
                ret[new_fake_state][action] = local_env.P[s_i][action]

    return ret


def cross_policies(policies: list, possible_states_counts: list):
    """Joint policy in a 'cross' matter.

    Args:
        policies: list of functions, function i is the policy for agent i.
        possible_states_counts: list of integers, each one is the number
            of states in the local view of agent i (including fake states for constraints).
    """
    if len(policies) != len(possible_states_counts):
        raise AssertionError("some went wrong!")

    n_agents = len(policies)

    def joint_policy(s):
        local_states = integer_to_vector(s, possible_states_counts, n_agents, lambda x: x)
        vector_joint_action = sum([integer_action_to_vector(policies[i](local_states[i]), 1)
                                   for i in range(n_agents)], ())
        joint_action = vector_action_to_integer(vector_joint_action)
        return joint_action

    return joint_policy


def best_joint_policy_under_constraint(env, constraints, n_agents):
    """Get a joint policy and its exptected sum of rewards."""
    policies = []
    total_reward = 0
    local_envs = []
    for i in range(n_agents):
        local_envs.append(get_local_view(env, [i]))
        # the n-th constraint for agent i takes place in the (env.nS+n+1)-th state of local env of agent i.
        agent_mask = constraints_to_mask(constraints[i], local_envs[i])
        local_envs[i].set_mask(agent_mask)

        r, p = value_iteration(local_envs[i])
        total_reward += r
        policies.append(p)  # solve as if agent i is alone

    possible_states_counts = [local_envs[i].nS for i in range(n_agents)]
    joint_policy = cross_policies(policies, possible_states_counts)
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


def sync_joint_policy(joint_policy, env, constraints, possible_states_counts):
    """Transform a joint policy from the space including constrains-states to the original one."""

    def joint_policy_synced(s):
        n_possible_locations = len(env.grid) * len(env.grid[0])
        # notice: here there is an assumption that s is a 'normal' state.
        # therefore the possible_states_counts parameter depends only on grid (and not on constraints as well)
        local_states = integer_to_vector(s, [n_possible_locations] * env.n_agents, env.n_agents, lambda x: x)
        for agent_idx in range(env.n_agents):
            # search for a constraint which includes i, s_i
            for ctr, (i, s_i, j, s_j, s_forbidden) in enumerate(constraints[agent_idx]):
                # this state is matching to a constraint.
                # sync the action of the conflicting agents
                # by letting the compromising agent to know he is in a special state.
                if local_states[i] == s_i and local_states[j] == s_j:
                    synced_vector_joint_state = local_states

                    synced_vector_joint_state = synced_vector_joint_state[:i] + (
                        n_possible_locations + ctr,) + synced_vector_joint_state[i + 1:]

                    # TODO: support more than one constraint under the same state
                    s = vector_to_integer(synced_vector_joint_state, possible_states_counts, lambda x: x)
                    return joint_policy(s)

        return joint_policy(s)

    return joint_policy_synced


def detect_conflict(env: MapfEnv, joint_policy: Callable[[int], int], n_agents: int):
    """Find a conflict between agents.

    A conflict is <i, s_i, j, s_j, s_ij> where:
    * i - index of first conflicting agent
    * s_i - local state which agent i was in before the clash
    * j - index of second conflicting agent
    * s_j - local state which agent j was in before the clash
    * s_ij - the shared state which both agents were in after their acting.
            One of the agent should avoid reaching this state when i is in s_i and j is in s_j.
    """
    visited_states = set()
    states_to_exapnd = [env.s]

    while len(states_to_exapnd) > 0:
        curr_expanded_state = states_to_exapnd.pop()
        visited_states.add(curr_expanded_state)
        joint_action = joint_policy(curr_expanded_state)
        for prob, next_state, reward, done in env.P[curr_expanded_state][joint_action]:
            if prob == 0:  # TODO: make gym_mapf not include transitions with probability 0
                continue

            next_state_vector = integer_state_to_vector(next_state, env.grid, n_agents)
            loc_count = Counter(next_state_vector)
            shared_locations = [loc for loc, counts in loc_count.items() if counts > 1]
            if len(shared_locations) != 0:  # clash between two agents
                first_agent = next_state_vector.index(shared_locations[0])
                second_agent = next_state_vector[first_agent + 1:].index(shared_locations[0]) + (first_agent + 1)

                # calculate the local states for each agent that with the current action got them here.
                vector_curr_expanded_state = integer_state_to_vector(curr_expanded_state, env.grid, env.n_agents)

                return (first_agent,
                        vector_state_to_integer(env.grid, (vector_curr_expanded_state[first_agent],)),
                        second_agent,
                        vector_state_to_integer(env.grid, (vector_curr_expanded_state[second_agent],)),
                        vector_state_to_integer(env.grid, (next_state_vector[first_agent],)))

            if next_state not in visited_states:
                states_to_exapnd.append(next_state)

    return None


def UCBS(env, n_agents):
    # TODO: problematic action can be problematic state to reach instead. find_conflict should solve this.
    curr_constraints = []
    curr_joint_reward, curr_joint_policy = best_joint_policy_under_constraint(env, curr_constraints, n_agents)
    search_tree = [(curr_joint_reward, curr_constraints, curr_joint_policy)]
    heapq.heapify(search_tree)

    while True:  # TODO: when to stop?
        conflict = detect_conflict(env, curr_joint_policy, n_agents)
        if not conflict:
            return curr_joint_reward, curr_joint_policy  # This is the best policy which has no conflicts.

        i, s_i, j, s_j, s_forbidden = conflict  # a conflict is where pi_i(s_i) and pi_j(s_j) might cause a collision

        # solve again but now agent i in s_i can't do pi_i[s_i] if j is in state s_j
        heapq.heappush(search_tree,
                       best_joint_policy_under_constraint(env,
                                                          curr_constraints[:i] +
                                                          (curr_constraints[i] +
                                                           (i, s_i, j, s_j, s_forbidden))
                                                          + curr_constraints[i + 1:],
                                                          n_agents))

        # now agent j is the one who compromises
        heapq.heappush(search_tree,
                       best_joint_policy_under_constraint(env,
                                                          curr_constraints[:j] +
                                                          (curr_constraints[j] +
                                                           (j, s_j, i, s_i, s_forbidden))
                                                          + curr_constraints[j + 1:],
                                                          n_agents))

        curr_joint_reward, curr_constraints, curr_joint_policy = search_tree.pop()
