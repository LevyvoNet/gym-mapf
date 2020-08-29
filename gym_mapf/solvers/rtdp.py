import numpy as np
import time
import math
import collections
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
    """Run a single iteration of RTDP.

    Args:
        policy (RtdpPolicy): the current policy (RTDP is an on-policy algorithm)
        info (Dict): optional for gathering information about the iteration - time, reward, special events, etc.

    Returns:
        float. The total reward of the episode.
    """
    s = policy.env.reset()
    done = False
    start = time.time()
    path = []
    total_reward = 0

    while not done:
        # time.sleep(0.1)
        # policy.env.render()
        # Choose greedy action (if there are several choose uniformly random)
        a = greedy_action(policy.env, s, policy.v, policy.gamma)
        # a = policy.act(s)
        path.append((s, a))

        # Do a bellman update
        bellman_update(policy, s, a)

        # Simulate the step and sample a new state
        s, r, done, _ = policy.env.step(a)
        total_reward += r

    # TODO: update backwards here using path variable

    # Write measures about that information
    info['time'] = time.time() - start
    info['n_moves'] = len(path)

    # Reset again just for safety
    policy.env.reset()

    return total_reward


def run_iterations_batch(policy: RtdpPolicy, iterations_batch_size: int, info: Dict):
    info['batch_iterations'] = []
    for i in range(iterations_batch_size):
        iter_info = {}
        info['batch_iterations'].append(iter_info)
        rtdp_single_iteration(policy, iter_info)


def rtdp_iterations_generator(heuristic_function: Callable[[MapfEnv], Callable[[int], float]],
                              gamma: float,
                              policy: RtdpPolicy,
                              env: MapfEnv,
                              info: Dict) -> Policy:
    info['iterations'] = []

    while True:
        info['iterations'].append({})
        iter_reward = rtdp_single_iteration(policy, info['iterations'][-1])
        yield iter_reward


def fixed_iterations_count_rtdp(heuristic_function: Callable[[MapfEnv], Callable[[int], float]],
                                gamma: float,
                                n_iterations: int,
                                env: MapfEnv,
                                info: Dict) -> Policy:
    # initialize V to an upper bound
    policy = RtdpPolicy(env, gamma, heuristic_function(env))

    for iter_count, reward in enumerate(rtdp_iterations_generator(heuristic_function, gamma, policy, env, info),
                                        start=1):
        if iter_count >= n_iterations:
            break

    return policy


def stop_when_no_improvement_between_batches_rtdp(heuristic_function: Callable[[MapfEnv], Callable[[int], float]],
                                                  gamma: float,
                                                  iterations_batch_size: int,
                                                  max_iterations: int,
                                                  env: MapfEnv,
                                                  info: Dict):
    def no_improvement_from_last_batch(policy: RtdpPolicy, iter_count: int):
        if iter_count % iterations_batch_size != 0:
            return False

        policy.policy_cache.clear()
        reward, _ = evaluate_policy(policy, 100, 1000)
        if reward == policy.env.reward_of_living * 1000:
            return False

        if not hasattr(policy, 'last_eval'):
            policy.last_eval = reward
            return False
        else:
            prev_eval = policy.last_eval
            policy.last_eval = reward
            return abs(policy.last_eval - prev_eval) / prev_eval <= 0.01

    # initialize V to an upper bound
    policy = RtdpPolicy(env, gamma, heuristic_function(env))

    # Run RTDP iterations
    for iter_count, reward in enumerate(rtdp_iterations_generator(heuristic_function, gamma, policy, env, info),
                                        start=1):
        # Stop when no improvement or when we have exceeded maximum number of iterations
        if no_improvement_from_last_batch(policy, iter_count) or iter_count >= max_iterations:
            break

    return policy
