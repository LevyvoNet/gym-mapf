import time
import json
from abc import ABCMeta, abstractmethod
from collections import Counter

import numpy as np

from gym_mapf.envs import integer_to_vector
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    integer_action_to_vector,
                                    vector_action_to_integer)
from gym_mapf.envs.utils import get_local_view


class Policy(metaclass=ABCMeta):
    def __init__(self, env: MapfEnv, gamma: float):
        self.env = env
        self.gamma = gamma

    @abstractmethod
    def act(self, s):
        """Return the policy action for a given state

        Args:
            s (int): a state of self environment

        Returns:
            int. The best action according to that policy.
        """

    @abstractmethod
    def dump_to_str(self):
        """Dump policy parameters to a string in a reproducible way

        Returns:
            str. string representation of the policy.
        """

    @staticmethod
    @abstractmethod
    def load_from_str(json_str: str) -> object:
        """Load policy from string

        Args:
            json_str (str): a string with a string representation dumped by the policy dump_to_json method.

        Returns:
            Policy. The policy represented by the given string.
        """


class CrossedPolicy(Policy):
    def __init__(self, env, gamma, policies):
        super().__init__(env, 1.0)
        self.policies = policies
        self.envs = [policy.env for policy in self.policies]

    def act(self, s):
        local_states = integer_to_vector(s, [env.nS for env in self.envs], len(self.envs), lambda x: x)
        vector_joint_action = sum(
            [integer_action_to_vector(self.policies[i].act(local_states[i]), self.envs[i].n_agents)
             for i in range(len(self.envs))], ())
        joint_action = vector_action_to_integer(vector_joint_action)
        return joint_action

    def dump_to_str(self):
        return json.dumps([policy.dump_to_str() for policy in self.policies])

    def load_from_str(json_str: str) -> object:
        raise NotImplementedError()


class Planner(metaclass=ABCMeta):
    def __init__(self):
        self.info = {}

    @abstractmethod
    def plan(self, env: MapfEnv, **kwargs) -> Policy:
        """Return a policy for a given MAPF env
        :param **kwargs:
        """

    @abstractmethod
    def dump_to_str(self):
        """Dump planner parameters to a string in a reproducible way

        Returns:
            str. string representation of the planner.
        """

    @staticmethod
    @abstractmethod
    def load_from_str(json_str: str) -> object:
        """Load planner from string

        Args:
            json_str (str): a string with a string representation dumped by the policy dump_to_json method.

        Returns:
            Planner. The policy represented by the given string.
        """


def print_path_to_state(path: dict, state: int, env: MapfEnv):
    curr_state = state
    print("final state: {}".format(env.state_to_locations(state)))
    while path[curr_state] is not None:
        curr_state, action = path[curr_state]
        print("state: {}, action: {}".format(env.state_to_locations(curr_state),
                                             integer_action_to_vector(action, env.n_agents)))


def detect_conflict(env: MapfEnv, joint_policy: Policy, **kwargs):
    """Find a conflict between agents.

    A conflict is <i, s_i, j, s_j, s_ij> where:
    * i - index of first conflicting agent
    * s_i - local state which agent i was in before the clash
    * j - index of second conflicting agent
    * s_j - local state which agent j was in before the clash
    * s_ij - the shared state which both agents were in after their acting.
            One of the agent should avoid reaching this state when i is in s_i and j is in s_j.
    """
    info = kwargs.get('info', {})
    start = time.time()
    visited_states = set()
    states_to_exapnd = [env.s]
    path = {env.s: None}
    aux_local_env = get_local_view(env, [0])

    while len(states_to_exapnd) > 0:
        curr_expanded_state = states_to_exapnd.pop()
        visited_states.add(curr_expanded_state)
        joint_action = joint_policy.act(curr_expanded_state)
        for prob, next_state, reward, done in env.P[curr_expanded_state][joint_action]:
            next_state_vector = env.state_to_locations(next_state)
            loc_count = Counter(next_state_vector)
            shared_locations = [loc for loc, counts in loc_count.items() if counts > 1]
            if len(shared_locations) != 0:  # clash between two agents
                # TODO: shouldn't I take care of every shared location instead of just the first one?
                first_agent = next_state_vector.index(shared_locations[0])
                second_agent = next_state_vector[first_agent + 1:].index(shared_locations[0]) + (first_agent + 1)

                # calculate the local states for each agent that with the current action got them here.
                vector_curr_expanded_state = env.state_to_locations(curr_expanded_state)

                info['detect_conflict_time'] = time.time() - start
                return (first_agent,
                        aux_local_env.locations_to_state((vector_curr_expanded_state[first_agent],)),
                        second_agent,
                        aux_local_env.locations_to_state((vector_curr_expanded_state[second_agent],)),
                        aux_local_env.locations_to_state((shared_locations[0],)))

            if next_state not in visited_states:
                states_to_exapnd.append(next_state)
                path[next_state] = (curr_expanded_state, joint_action)

    info['detect_conflict_time'] = time.time() - start
    return None


def might_conflict(clash_reward, transitions):
    for prob, new_state, reward, done in transitions:
        if reward == clash_reward and done:
            # This is a conflict transition
            return True

    return False


def safe_actions(env: MapfEnv, s):
    return [a for a in range(env.nA)
            if not might_conflict(env.reward_of_clash, env.P[s][a])]


def solve_independently_and_cross(env, agent_groups, low_level_planner: Planner, **kwargs):
    info = kwargs.get('info', {})
    start = time.time()  # TODO: use a decorator for updating info with time measurement
    local_envs = [get_local_view(env, group) for group in agent_groups]

    policies = []
    for group, local_env in zip(agent_groups, local_envs):
        info[f'{group}'] = {}
        policy = low_level_planner.plan(local_env, **{'info': info[f'{group}']})
        policies.append(policy)

    joint_policy = CrossedPolicy(env, 1.0, policies)

    end = time.time()
    info['best_joint_policy_time'] = end - start

    return joint_policy


def render_states(env, states):
    s_initial = env.s
    for state in states:
        env.s = state
        print(state)
        env.render()

    env.s = s_initial


class TabularValueFunctionPolicy(Policy):
    def __init__(self, env, gamma):
        super().__init__(env, gamma)
        self.v = np.zeros(env.nS)
        self.policy_cache = {}

    def act(self, s):
        if s in self.policy_cache:
            return self.policy_cache[s]

        possible_actions_from_state = safe_actions(self.env, s)
        q_sa = np.zeros(len(possible_actions_from_state))
        for a_idx in range(len(possible_actions_from_state)):
            a = possible_actions_from_state[a_idx]
            for next_sr in self.env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a_idx] += (p * (r + self.gamma * self.v[s_]))

        best_action = possible_actions_from_state[np.argmax(q_sa)]
        self.policy_cache[s] = best_action
        return best_action

    def dump_to_str(self):
        raise NotImplementedError()

    def load_from_str(json_str: str) -> object:
        raise NotImplementedError()
