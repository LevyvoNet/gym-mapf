import numpy as np
import time
import math
from typing import Callable

from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.solvers.vi import prioritized_value_iteration
from gym_mapf.solvers.utils import Planner, Policy, TabularValueFunctionPolicy, get_local_view
from gym_mapf.solvers.utils import render_states
from gym_mapf.envs.mapf_env import integer_action_to_vector


def greedy_action(env: MapfEnv, s, v, gamma):
    action_values = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            if reward == env.reward_of_clash and done:
                action_values[a] = -math.inf
                break

            action_values[a] += prob * (reward + (gamma * v[next_state]))

    # # for debug
    # for i in range(env.nA):
    #     print(f'{integer_action_to_vector(i, env.n_agents)}: {action_values[i]}')

    max_value = np.max(action_values)
    return np.random.choice(np.argwhere(action_values == max_value).flatten())


def rtdp(env: MapfEnv, heuristic_function: Callable[[int], float], n_iterations: int, gamma: float, **kwargs):
    info = kwargs.get('info', {})

    # initialize V to an upper bound
    # TODO: use lazy evaluation for v
    v = np.full(env.nS, 0.0)
    for s in range(env.nS):
        v[s] = heuristic_function(s)

    # follow the greedy policy, for each transition do a bellman update on V
    # import ipdb
    # ipdb.set_trace()
    for i in range(n_iterations):
        env.reset()
        s = env.s
        done = False
        start = time.time()
        n_moves = 0
        while not done:
            # env.render()
            a = greedy_action(env, s, v, gamma)
            # print(f'action {integer_action_to_vector(a, env.n_agents)} chosen')
            # time.sleep(1)
            new_v_s = sum([prob * (reward + gamma * v[next_state])
                           for prob, next_state, reward, done in env.P[s][a]])
            v[s] = new_v_s

            # simulate the step and sample a new state
            # import ipdb
            # ipdb.set_trace()
            s, r, done, _ = env.step(a)
            n_moves += 1

        # iteration finished
        print(f"iteration {i + 1} took {time.time() - start} seconds for {n_moves} moves, final reward: {r}")

    env.reset()
    policy = TabularValueFunctionPolicy(env, gamma)
    policy.v = v

    return policy


def manhattan_heuristic(env: MapfEnv):
    def heuristic_function(s):
        locations = env.state_to_locations(s)
        manhatten_distance = [
            abs(locations[i][0] - env.agents_goals[i][0]) + abs(locations[i][1] - env.agents_goals[i][1])
            for i in range(env.n_agents)]

        # MapfEnv reward is Makespan oriented
        return env.reward_of_goal + env.reward_of_living * max(manhatten_distance)

    return heuristic_function


def prioritized_value_iteration_heuristic(env: MapfEnv) -> Callable[[int], float]:
    local_envs = [get_local_view(env, [i]) for i in range(env.n_agents)]
    local_v = [prioritized_value_iteration(local_env, {}, 1.0) for local_env in local_envs]

    def heuristic_function(s):
        locations = env.state_to_locations(s)
        local_states = [local_envs[i].locations_to_state((locations[i],)) for i in range(env.n_agents)]
        # at the current, the MapfEnv reward is makespan oriented
        return min([local_v[i][local_states[i]] for i in range(env.n_agents)])

    return heuristic_function


class RtdpPlanner(Planner):
    def __init__(self, heuristic_function: Callable[[MapfEnv], Callable[[int], float]], n_iterations: int,
                 gamma: float):
        self.heuristic_function = heuristic_function
        self.n_iterations = n_iterations
        self.gamma = gamma

    def plan(self, env: MapfEnv, **kwargs) -> Policy:
        return rtdp(env, self.heuristic_function(env), self.n_iterations, self.gamma)

    def dump_to_str(self):
        pass

    @staticmethod
    def load_from_str(json_str: str) -> object:
        pass
