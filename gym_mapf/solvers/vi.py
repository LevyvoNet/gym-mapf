import time
import numpy as np
import math
import json

from gym_mapf.solvers.utils import safe_actions, TabularValueFunctionPolicy, Planner, Policy
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


class ValueIterationPlanner(Planner):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def plan(self, env: MapfEnv, **kwargs) -> TabularValueFunctionPolicy:
        """ Value-iteration algorithm"""
        self.info['converged'] = False
        self.info['n_iterations'] = 0
        start = time.time()  # TODO: use a decorator for updating info with time measurement
        gamma = kwargs.get('gamma', 1.0)
        v = np.zeros(env.nS)  # initialize value-function
        max_iterations = 1000
        eps = 1e-2
        s_count = 0
        real_start = time.time()
        for i in range(max_iterations):
            prev_v = np.copy(v)
            start = time.time()
            for s in range(env.nS):
                q_sa = []
                for a in range(env.nA):
                    q_sa_a = 0
                    for p, s_, r, done in env.P[s][a]:
                        if r == env.reward_of_clash and done:
                            # This is a dangerous action which might get to conflict
                            q_sa_a = -math.inf
                            break
                        q_sa_a += p * (r + prev_v[s_])

                    q_sa.append(q_sa_a)

                v[s] = max(q_sa)
                s_count += 1

            # debug print
            # if i % 10 == 0:
            #     print(v)

            print(f'VI: iteration {i + 1} took {time.time() - start} seconds')

            self.info['n_iterations'] = i + 1
            if np.sum(np.fabs(prev_v - v)) <= eps:
                # debug print
                print('value iteration converged at iteration# %d.' % (i + 1))
                self.info['converged'] = True
                break

        policy = TabularValueFunctionPolicy(env, self.gamma)
        policy.v = v

        end = time.time()
        self.info['VI_time'] = end - start

        return policy

    def dump_to_str(self):
        return json.dumps({'gamma': self.gamma})

    @staticmethod
    def load_from_str(json_str: str) -> object:
        json_obj = json.loads(json_str)
        return ValueIterationPlanner(json_obj['gamma'])


class PrioritizedValueIterationPlanner(Planner):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def plan(self, env: MapfEnv, **kwargs) -> TabularValueFunctionPolicy:
        self.info['converged'] = False
        self.info['n_iterations'] = 0
        start = time.time()  # TODO: use a decorator for updating info with time measurement
        v = np.zeros(env.nS)  # initialize value-function
        max_iterations = 100000
        eps = 1e-2
        q_sa_a = 0
        layers = get_layers(env)
        for i in range(max_iterations):
            prev_v = np.copy(v)
            start = time.time()
            for layer in layers:
                for s in layer:
                    q_sa = []
                    for a in range(env.nA):
                        q_sa_a = 0
                        for p, s_, r, done in env.P[s][a]:
                            if r == env.reward_of_clash and done:
                                # This is a dangerous action which might get to conflict
                                q_sa_a = -math.inf
                                break
                            q_sa_a += p * (r + v[s_])

                        q_sa.append(q_sa_a)

                    v[s] = max(q_sa)

            # debug print
            # if i % 10 == 0:
            #     print(v)
            print(f'PVI: iteration {i + 1} took {time.time() - start} seconds')

            self.info['n_iterations'] = i + 1
            if np.sum(np.fabs(prev_v - v)) <= eps:
                # debug print
                print('prioritized value iteration converged at iteration# %d.' % (i + 1))
                self.info['converged'] = True
                break

        policy = TabularValueFunctionPolicy(env, self.gamma)
        policy.v = v

        end = time.time()
        self.info['prioritized_VI_time'] = end - start
        return policy

    def dump_to_str(self):
        return json.dumps({'gamma': self.gamma})

    @staticmethod
    def load_from_str(json_str: str) -> object:
        json_obj = json.loads(json_str)
        return ValueIterationPlanner(json_obj['gamma'])
