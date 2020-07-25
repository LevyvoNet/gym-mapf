import numpy as np
import time
import math
from typing import Callable, Dict

from gym_mapf.envs.mapf_env import MapfEnv, function_to_get_item_of_object, integer_action_to_vector
from gym_mapf.solvers.vi import PrioritizedValueIterationPlanner
from gym_mapf.solvers.utils import Planner, Policy, TabularValueFunctionPolicy, get_local_view


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


# TODO: why is it so important to get a random greedy action (instead of just the first index?).
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


def prioritized_value_iteration_heuristic(env: MapfEnv) -> Callable[[int], float]:
    local_envs = [get_local_view(env, [i]) for i in range(env.n_agents)]
    pvi_planner = PrioritizedValueIterationPlanner(1.0)
    local_v = [(pvi_planner.plan(local_env, {})).v for local_env in local_envs]

    def heuristic_function(s):
        locations = env.state_to_locations(s)
        local_states = [local_envs[i].locations_to_state((locations[i],)) for i in range(env.n_agents)]
        # at the current, the MapfEnv reward is makespan oriented
        return min([local_v[i][local_states[i]] for i in range(env.n_agents)])

    return heuristic_function


class RtdpPlanner(Planner):
    def __init__(self, heuristic_function: Callable[[MapfEnv], Callable[[int], float]], n_iterations: int,
                 gamma: float):
        super().__init__()
        self.heuristic_function = heuristic_function
        self.n_iterations = n_iterations
        self.gamma = gamma

    def plan(self, env: MapfEnv, info: Dict, **kwargs) -> Policy:
        info['iterations'] = []

        # initialize V to an upper bound
        policy = RtdpPolicy(env, self.gamma, self.heuristic_function(env))

        # follow the greedy policy, for each transition do a bellman update on V
        for i in range(self.n_iterations):
            env.reset()
            s = env.s
            done = False
            start = time.time()
            path = []
            while not done:
                # env.render()
                a = greedy_action(env, s, policy.v, self.gamma)
                # a = policy.act(s)
                path.append((s, a))
                # print(f'action {integer_action_to_vector(a, env.n_agents)} chosen')
                # time.sleep(1)
                new_v_s = sum([prob * (reward + self.gamma * policy.v[next_state])
                               for prob, next_state, reward, done in env.P[s][a]])
                policy.v_partial_table[s] = new_v_s

                # simulate the step and sample a new state
                s, r, done, _ = env.step(a)

            info['iterations'].append({
                'n_moves': len(path),
                'time': time.time() - start
            })
            # iteration finished
            # n_moves = len(path)
            # print(f"iteration {i + 1} took {time.time() - start} seconds for {n_moves} moves, final reward: {r}")

        env.reset()

        return policy

    def dump_to_str(self):
        pass

    @staticmethod
    def load_from_str(json_str: str) -> object:
        pass
