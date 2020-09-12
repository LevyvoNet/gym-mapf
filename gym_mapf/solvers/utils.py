import time
import json
from abc import ABCMeta, abstractmethod
from collections import Counter
from typing import Dict, Callable

import numpy as np

from gym_mapf.envs import integer_to_vector
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    integer_action_to_vector,
                                    vector_action_to_integer)
from gym_mapf.envs.utils import get_local_view, manhattan_distance


class Policy(metaclass=ABCMeta):
    def __init__(self, env: MapfEnv, gamma: float):
        # TODO: deep copy env, don't just copy the reference
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
    def __init__(self, env, policies, agents_groups):
        super().__init__(env, 1.0)  # This does not matter
        self.policies = policies
        self.envs = [policy.env for policy in self.policies]
        self.agents_groups = agents_groups

    def act(self, s):
        agent_locations = self.env.state_to_locations(s)
        agent_to_action = {}

        for i in range(len(self.agents_groups)):
            local_env_agent_locations = sum([(agent_locations[agent],)
                                             for agent in self.agents_groups[i]], ())

            local_env_agent_state = self.envs[i].locations_to_state(local_env_agent_locations)

            local_action = self.policies[i].act(local_env_agent_state)

            local_vector_action = integer_action_to_vector(local_action, self.envs[i].n_agents)
            for j, agent in enumerate(self.agents_groups[i]):
                agent_to_action[agent] = local_vector_action[j]
        joint_action_vector = tuple([action for agent, action in sorted(agent_to_action.items())])
        joint_action = vector_action_to_integer(joint_action_vector)

        return joint_action

    def dump_to_str(self):
        return json.dumps([policy.dump_to_str() for policy in self.policies])

    def load_from_str(json_str: str) -> object:
        raise NotImplementedError()


def print_path_to_state(path: dict, state: int, env: MapfEnv):
    curr_state = state
    print("final state: {}".format(env.state_to_locations(state)))
    while path[curr_state] is not None:
        curr_state, action = path[curr_state]
        print("state: {}, action: {}".format(env.state_to_locations(curr_state),
                                             integer_action_to_vector(action, env.n_agents)))


def detect_conflict(env: MapfEnv, joint_policy: Policy, **kwargs):
    """Find a conflict between agents.

    A conflict is ((i,,s_i,new_s_i), (j,s_j,new_s_j)) where:
    * i - index of first conflicting agent
    * s_i - local state which agent i was in before the clash
    * new_s_i = local state which agent i was in after the clash
    * j - index of second conflicting agent
    * s_j - local state which agent j was in before the clash
    * new_s_j - local state which agent j was in after the clash
    """
    info = kwargs.get('info', {})
    start = time.time()
    visited_states = set()
    env.reset()
    states_to_expand = [env.s]
    path = {env.s: None}
    aux_local_env = get_local_view(env, [0])

    while len(states_to_expand) > 0:
        curr_expanded_state = states_to_expand.pop()
        visited_states.add(curr_expanded_state)
        joint_action = joint_policy.act(curr_expanded_state)
        # print(f'{len(states_to_expand)} to expand, {len(visited_states)} already expanded, total {env.nS} states')
        for prob, next_state, reward, done in env.P[curr_expanded_state][joint_action]:
            # next_state_vector = env.state_to_locations(next_state)
            # loc_count = Counter(next_state_vector)
            # shared_locations = [loc for loc, counts in loc_count.items() if counts > 1]
            if done and reward == env.reward_of_clash:  # clash between two agents
                # TODO: shouldn't I take care of every shared location instead of just the first one?
                next_state_vector = env.state_to_locations(next_state)
                loc_count = Counter(next_state_vector)
                shared_locations = [loc for loc, counts in loc_count.items() if counts > 1]
                if len(shared_locations) > 0:
                    # classical clash
                    first_agent = next_state_vector.index(shared_locations[0])
                    second_agent = next_state_vector[first_agent + 1:].index(shared_locations[0]) + (first_agent + 1)
                else:
                    # switch between two agents
                    curr_state_vector = env.state_to_locations(curr_expanded_state)
                    shared_locations = [loc for loc in next_state_vector if loc in curr_state_vector]
                    for shared_loc in shared_locations:
                        # check if this is just an agent which remained in place
                        first_agent = curr_state_vector.index(shared_loc)
                        second_agent = next_state_vector.index(shared_loc)
                        if first_agent != second_agent:
                            # these agents made a switch
                            return (
                                (
                                    first_agent,
                                    aux_local_env.locations_to_state((curr_state_vector[first_agent],)),
                                    aux_local_env.locations_to_state((next_state_vector[first_agent],))
                                ),
                                (
                                    second_agent,
                                    aux_local_env.locations_to_state((curr_state_vector[second_agent],)),
                                    aux_local_env.locations_to_state((next_state_vector[second_agent],))
                                )
                            )

                    assert False, "Something is wrong - MapfEnv had a conflict but there isn't"

                # calculate the local states for each agent that with the current action got them here.
                vector_curr_expanded_state = env.state_to_locations(curr_expanded_state)

                info['detect_conflict_time'] = round(time.time() - start, 2)
                return (
                    (
                        first_agent,
                        aux_local_env.locations_to_state((vector_curr_expanded_state[first_agent],)),
                        aux_local_env.locations_to_state((shared_locations[0],))
                    ),
                    (
                        second_agent,
                        aux_local_env.locations_to_state((vector_curr_expanded_state[second_agent],)),
                        aux_local_env.locations_to_state((shared_locations[0],))
                    )
                )

            if next_state not in visited_states:
                states_to_expand.append(next_state)
                path[next_state] = (curr_expanded_state, joint_action)

    info['detect_conflict_time'] = round(time.time() - start, 2)
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


def solve_independently_and_cross(env, agent_groups, low_level_planner: Callable[[MapfEnv, Dict], Policy], info: Dict):
    """Solve the MDP MAPF for the local views of the given agent groups

    Args:
        agent_groups (list): a list of lists, each list is a group of agents.
        low_level_planner ((MapfEnv)->Policy): a low level planner to solve the local envs with.
        info (dict): information to update during the solving
    """
    start = time.time()  # TODO: use a decorator for updating info with time measurement
    local_envs = [get_local_view(env, group) for group in agent_groups]

    policies = []
    for group, local_env in zip(agent_groups, local_envs):
        info[f'{group}'] = {}
        policy = low_level_planner(local_env, info[f'{group}'])
        policies.append(policy)

    joint_policy = CrossedPolicy(env, policies, agent_groups)

    end = time.time()
    info['best_joint_policy_time'] = round(end - start, 2)

    return joint_policy


def render_states(env, states):
    s_initial = env.s
    for state in states:
        env.s = state
        print(state)
        env.render()

    env.s = s_initial


def evaluate_policy(policy: Policy, n_episodes: int, max_steps: int):
    total_reward = 0
    clashed = False
    for i in range(n_episodes):
        policy.env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            # # debug print
            # print(f'steps={steps}')
            # policy.env.render()
            # time.sleep(1)
            new_state, reward, done, info = policy.env.step(policy.act(policy.env.s))
            total_reward += reward
            steps += 1
            if reward == policy.env.reward_of_clash and done:
                print("clash happened!!!")
                clashed = True

    policy.env.reset()
    return total_reward / n_episodes, clashed


class ValueFunctionPolicy(Policy):
    def __init__(self, env, gamma):
        super().__init__(env, gamma)
        self.v = []
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
