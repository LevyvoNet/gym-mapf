import numpy as np
import time
import math
from typing import Callable, Dict

from gym_mapf.envs.mapf_env import MapfEnv, function_to_get_item_of_object, integer_action_to_vector
from gym_mapf.solvers.vi import prioritized_value_iteration
from gym_mapf.solvers.utils import Policy, TabularValueFunctionPolicy, get_local_view, evaluate_policy


class RtdpPolicy(TabularValueFunctionPolicy):
    def __init__(self, env, gamma, heuristic):
        super().__init__(env, gamma)
        self.v_partial_table = {}
        # Now this v behaves like a full numpy array
        self.v = function_to_get_item_of_object(self._get_value)
        self.heuristic = heuristic

    def _get_value(self, s):
        if s in self.v_partial_table:
            return self.v_partial_table[s]

        value = self.heuristic(s)
        self.v_partial_table[s] = value
        return value

    def dump_to_str(self):
        pass

    def load_from_str(json_str: str) -> object:
        pass


# TODO: Is really important to get a random greedy action (instead of just the first index?).
#  I wish I could delete this function and just use `policy.act(s)` instead
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


def manhattan_heuristic(env: MapfEnv):
    def heuristic_function(s):
        locations = env.state_to_locations(s)
        manhatten_distance = [
            abs(locations[i][0] - env.agents_goals[i][0]) + abs(locations[i][1] - env.agents_goals[i][1])
            for i in range(env.n_agents)]

        # MapfEnv reward is Makespan oriented
        return env.reward_of_goal + env.reward_of_living * max(manhatten_distance)

    return heuristic_function


def prioritized_value_iteration_heuristic(gamma: float, env: MapfEnv) -> Callable[[int], float]:
    local_envs = [get_local_view(env, [i]) for i in range(env.n_agents)]
    local_v = [(prioritized_value_iteration(gamma, local_env, {})).v for local_env in local_envs]

    def heuristic_function(s):
        locations = env.state_to_locations(s)
        local_states = [local_envs[i].locations_to_state((locations[i],)) for i in range(env.n_agents)]
        # at the current, the MapfEnv reward is makespan oriented
        return min([local_v[i][local_states[i]] for i in range(env.n_agents)])

    return heuristic_function


def bellman_update(policy: RtdpPolicy, s: int, a: int):
    new_v_s = sum([prob * (reward + policy.gamma * policy.v[next_state])
                   for prob, next_state, reward, done in policy.env.P[s][a]])
    policy.v_partial_table[s] = new_v_s


def rtdp_single_iteration(policy: RtdpPolicy, info: Dict):
    s = policy.env.reset()
    done = False
    start = time.time()
    path = []

    while not done:
        # Choose greedy action (if there are several choose uniformly random)
        a = greedy_action(policy.env, s, policy.v, policy.gamma)
        path.append((s, a))

        # Do a bellman update
        bellman_update(policy, s, a)

        # Simulate the step and sample a new state
        s, r, done, _ = policy.env.step(a)

    # TODO: update backwards here using path variable

    # Write measures about that information
    info['time'] = time.time() - start
    info['n_moves'] = len(path)

    # Reset again just for safety
    policy.env.reset()


def run_iterations_batch(policy: RtdpPolicy, iterations_batch_size: int, info: Dict):
    info['batch_iterations'] = []
    for _ in range(iterations_batch_size):
        iter_info = {}
        info['batch_iterations'].append(iter_info)
        rtdp_single_iteration(policy, iter_info)


def rtdp(heuristic_function: Callable[[MapfEnv], Callable[[int], float]],
         gamma: float,
         should_stop: Callable[[Policy], bool],
         iterations_batch_size: int,
         max_iterations: int,
         env: MapfEnv,
         info: Dict):
    info['batches'] = []
    iterations_count = 0

    # initialize V to an upper bound
    policy = RtdpPolicy(env, gamma, heuristic_function(env))

    # Run a batch of iterations for the first time
    batch_info = {}
    info['batches'].append(batch_info)
    run_iterations_batch(policy, iterations_batch_size, batch_info)
    iterations_count += iterations_batch_size

    while not should_stop(policy) and iterations_count < max_iterations:
        batch_info = {}
        info['batches'].append(batch_info)
        run_iterations_batch(policy, iterations_batch_size, batch_info)
        iterations_count += iterations_batch_size

    return policy
