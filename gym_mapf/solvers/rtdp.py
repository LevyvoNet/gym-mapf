import numpy as np
import time
import math
from typing import Callable

from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.solvers.utils import TabularValueFunctionPolicy


def greedy_action(env: MapfEnv, s, v, gamma):
    action_values = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            if reward == env.reward_of_clash and done:
                action_values[a] = -math.inf
                break

            action_values[a] += prob * (reward + (gamma * v[next_state]))

    max_value = np.max(action_values)
    return np.random.choice(np.argwhere(action_values == max_value).flatten())


def rtdp(env: MapfEnv, heuristic_function: Callable[[int], float], n_iterations: int, **kwargs):
    info = kwargs.get('info', {})
    gamma = kwargs.get('gamma', 1.0)

    # initialize V to an upper bound
    v = np.full(env.nS, 0.0)
    for s in range(env.nS):
        v[s] = heuristic_function(s)

    # follow the greedy policy, for each transition do a bellman update on V
    for i in range(n_iterations):
        env.reset()
        s = env.s
        done = False
        start = time.time()
        n_moves = 0
        while not done:
            a = greedy_action(env, s, v, gamma)
            v[s] = 0
            for prob, next_state, reward, done in env.P[s][a]:
                v[s] += prob * (reward + gamma * v[next_state])

            # simulate the step and sample a new state
            s, r, done, _ = env.step(a)
            n_moves += 1

        # iteration finished
        print(f"iteration {i + 1} took {time.time() - start} seconds for {n_moves} moves, final reward: {r}")

    env.reset()
    policy = TabularValueFunctionPolicy(env, gamma)
    policy.v = v

    return policy
