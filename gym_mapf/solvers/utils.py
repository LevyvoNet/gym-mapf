from collections import Counter
from typing import Callable

from gym_mapf.envs import integer_to_vector
from gym_mapf.envs.mapf_env import integer_action_to_vector, vector_action_to_integer, MapfEnv, integer_state_to_vector, \
    vector_state_to_integer


def cross_policies(policies: list, envs: list):
    """Joint policy in a 'cross' matter.

    Args:
        policies: list of functions, function i is the policy for agent i.
        envs: list of matching envs for each of the policies
    """
    if len(policies) != len(envs):
        raise AssertionError("some went wrong!")

    n_envs = len(envs)

    def joint_policy(s):
        local_states = integer_to_vector(s, [env.nS for env in envs], n_envs, lambda x: x)
        vector_joint_action = sum([integer_action_to_vector(policies[i](local_states[i]), envs[i].n_agents)
                                   for i in range(n_envs)], ())
        joint_action = vector_action_to_integer(vector_joint_action)
        return joint_action

    return joint_policy


def detect_conflict(env: MapfEnv, joint_policy: Callable[[int], int]):
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

            next_state_vector = integer_state_to_vector(next_state, env.grid, env.n_agents)
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


def might_conflict(clash_reward, transitions):
    for prob, new_state, reward, done in transitions:
        if reward == clash_reward and done:
            # This is a conflict transition
            return True

    return False


def safe_actions(env, s):
    return [a for a in range(env.nA)
            if not might_conflict(env.reward_of_clash, env.P[s][a])]
