import time
import numpy as np

from gym_mapf.solvers.utils import safe_actions
from gym_mapf.envs.mapf_env import MapfEnv


def extract_policy(v, env, gamma=1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        possible_actions_from_state = safe_actions(env, s)
        q_sa = np.zeros(len(possible_actions_from_state))
        for a_idx in range(len(possible_actions_from_state)):
            a = possible_actions_from_state[a_idx]
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a_idx] += (p * (r + gamma * v[s_]))
        policy[s] = possible_actions_from_state[np.argmax(q_sa)]
    return policy


def value_iteration(env, info, gamma=1.0):
    """ Value-iteration algorithm"""
    info['converged'] = False
    info['n_iterations'] = 0
    v = np.zeros(env.nS)  # initialize value-function
    max_iterations = 100000
    eps = 1e-2
    for i in range(max_iterations):
        prev_v = np.copy(v)
        start = time.time()
        for s in range(env.nS):
            q_sa = [sum([p * (r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in safe_actions(env, s)]
            v[s] = max(q_sa)

        # debug print
        # if i % 10 == 0:
        #     print(v)
        print(f'VI: iteration {i+1} took {time.time() - start} seconds')

        info['n_iterations'] = i + 1
        if np.sum(np.fabs(prev_v - v)) <= eps:
            # debug print
            print('value iteration converged at iteration# %d.' % (i + 1))
            info['converged'] = True
            break

    return v


def render_states(env, states):
    s_initial = env.s
    for state in states:
        env.s = state
        print(state)
        env.render()

    env.s = s_initial


def get_layers(env):
    layers = []
    visited_states = set()
    iter_states = set(env.predecessors(env.locations_to_state(env.agents_goals)))
    next_iter_states = set(iter_states)
    while len(visited_states) < env.nS:
        iter_states = set(next_iter_states)
        next_iter_states = set()
        for s in iter_states:
            visited_states.add(s)
            next_iter_states = next_iter_states.union(env.predecessors(s))

        next_iter_states = next_iter_states.difference(visited_states)

        layers.append(iter_states)

    return layers


def prioritized_value_iteration(env: MapfEnv, info, gamma=1.0):
    info['converged'] = False
    info['n_iterations'] = 0
    v = np.zeros(env.nS)  # initialize value-function
    max_iterations = 100000
    eps = 1e-2
    layers = get_layers(env)
    for i in range(max_iterations):
        prev_v = np.copy(v)
        start = time.time()
        for layer in layers:
            for s in layer:
                q_sa = [sum([p * (r + v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in safe_actions(env, s)]
                v[s] = max(q_sa)

        # debug print
        # if i % 10 == 0:
        #     print(v)
        print(f'PVI: iteration {i+1} took {time.time() - start} seconds')

        info['n_iterations'] = i + 1
        if np.sum(np.fabs(prev_v - v)) <= eps:
            # debug print
            print('prioritized value iteration converged at iteration# %d.' % (i + 1))
            info['converged'] = True
            break

    return v


def value_iteration_planning(env, **kwargs):
    """Get optimal policy derived from value iteration and its expected reward"""
    info = kwargs.get('info', {})
    start = time.time()  # TODO: use a decorator for updating info with time measurement
    gamma = 1.0
    v = value_iteration(env, info, gamma)
    policy_table = extract_policy(v, env, gamma)

    def policy_int_output(s):
        return int(policy_table[s])

    end = time.time()
    info['VI_time'] = end - start
    return v[env.s], policy_int_output


def prioritized_value_iteration_planning(env, **kwargs):
    info = kwargs.get('info', {})
    start = time.time()  # TODO: use a decorator for updating info with time measurement
    gamma = 1.0
    v = prioritized_value_iteration(env, info, gamma)
    policy_table = extract_policy(v, env, gamma)

    def policy_int_output(s):
        return int(policy_table[s])

    end = time.time()
    info['prioritized_VI_time'] = end - start
    return v[env.s], policy_int_output
