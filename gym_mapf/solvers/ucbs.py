import numpy as np
import heapq
from collections import Counter

from gym_mapf.envs.utils import get_local_view
from gym_mapf.envs.mapf_env import (vector_state_to_integer,
                                    vector_action_to_integer,
                                    integer_state_to_vector,
                                    integer_action_to_vector)
from gym_mapf.solvers.value_iteration_agent import ValueIterationAgent


def add_constraint(*args):
    pass


def value_iteration(env):
    """Get optimal policy derived from value iteration and its expected reward"""
    vi_agent = ValueIterationAgent()
    vi_agent.train(env)

    return vi_agent.optimal_v[env.s], vi_agent.policy


def find_best_policies(env, n_agents):
    """Get policies and their correspond expected reward.

    For example:
    [(2, policy_1), (5, policy_2),...,(6, policy_2) ]

    Each element in the list is a two dimensional tuple. The first element of the tuple
    is the value of the optimal policy for agent i, and the second
    one is the policy itself (function from state to action).
    """
    policies = []
    total_reward = 0
    for i in range(n_agents):
        r, p = value_iteration(get_local_view(env, [i]))
        total_reward += r
        policies.append(p)  # solve as if agent i is alone

    return total_reward, policies


def find_conflict(env, joint_policy, n_agents):
    visited_states = set()
    states_to_exapnd = [env.s]

    while not len(states_to_exapnd) == 0:
        curr_expanded_state = states_to_exapnd.pop()
        visited_states.add(curr_expanded_state)
        vector_joint_state = integer_state_to_vector(curr_expanded_state, env.grid, n_agents)
        local_states = [vector_state_to_integer(env.grid, (vector_joint_state[i],))
                        for i in range(n_agents)]
        vector_joint_action = sum([integer_action_to_vector(joint_policy[i][local_states[i]], 1)
                                   for i in range(n_agents)], ())
        joint_action = vector_action_to_integer(vector_joint_action)
        for prob, next_state, reward, done in env.P[curr_expanded_state][joint_action]:
            if prob == 0:
                continue

            next_state_vector = integer_state_to_vector(next_state, env.grid, n_agents)
            loc_count = Counter(next_state_vector)
            shared_locations = [loc for loc, counts in loc_count.items() if counts > 1]
            if len(shared_locations) != 0:  # clash between two agents
                first_agent = next_state_vector.index(shared_locations[0])
                second_agent = next_state_vector[first_agent + 1:].index(shared_locations[0]) + (first_agent + 1)

                return first_agent, vector_joint_state[first_agent], second_agent, vector_joint_state[second_agent]

            if next_state not in visited_states:
                states_to_exapnd.append(next_state)

    return None

    # here visited states has all the possible states from the evaluated policy.


def UCBS(env, n_agents):
    curr_joint_reward, curr_joint_policy = find_best_policies(env, n_agents)
    search_tree = [(curr_joint_reward, curr_joint_policy)]
    heapq.heapify(search_tree)

    while True:  # TODO: when to stop?
        # finding_conflict is simliar to policy evaluation but instead of computing the expected reward for each state we are comuting the probabilty to reach it.
        conflict = find_conflict(env, curr_joint_policy, n_agents)
        if not conflict:
            return curr_joint_reward, curr_joint_policy  # This is the best policy which has no conflicts.

        i, s_i, j, s_j = conflict  # a conflict is where pi_i(s_i) and pi_j(s_j) might cause a collision

        new_mdp_a = add_constraint(env, i, s_i, curr_joint_policy[i],
                                   j, s_j)  # the constraint is that i must not do pi_i(s_i) if j is in state s_j
        heapq.heappush(search_tree, find_best_policies(new_mdp_a, n_agents))

        new_mdp_b = add_constraint(env, j, s_j, curr_joint_policy[j],
                                   i, s_i)  # now agent j is the one who compromises
        heapq.heappush(search_tree, find_best_policies(new_mdp_b, n_agents))

        r, p = search_tree.pop()  # a not exapnded node which have the best joint policy (the highest joint expected reward)
