import time
import numpy as np
import math
import json
from typing import Dict

from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.solvers.utils import Planner, Policy, TabularValueFunctionPolicy


def one_step_lookahead(env, state, V, discount_factor=1.0):
    """
    Helper function to  calculate state-value function

    Arguments:
        env: openAI GYM Enviorment object
        state: state to consider
        V: Estimated Value for each state. Vector of length nS
        discount_factor: MDP discount factor

    Return:
        action_values: Expected value of each action in a state. Vector of length nA
    """
    # initialize vector of action values
    action_values = np.zeros(env.nA)

    # loop over the actions we can take in an enviorment
    for action in range(env.nA):
        # loop over the P_sa distribution.
        for probablity, next_state, reward, done in env.P[state][action]:
            # if we are in state s and take action a. then sum over all the possible states we can land into.
            if reward == env.reward_of_clash and done:
                action_values[action] = -math.inf
                break

            action_values[action] += probablity * (reward + (discount_factor * V[next_state]))

    return action_values


def update_policy(env, policy, V, discount_factor):
    """
    Helper function to update a given policy based on given value function.

    Arguments:
        env: openAI GYM Environment object.
        policy: policy to update.
        V: Estimated Value for each state. Vector of length nS.
        discount_factor: MDP discount factor.
    Return:
        policy: Updated policy based on the given state-Value function 'V'.
    """
    for state in range(env.nS):
        # for a given state compute state-action value.
        action_values = one_step_lookahead(env, state, V, discount_factor)

        # choose the action which maximize the state-action value.
        policy[state] = np.argmax(action_values)

    return policy


def policy_eval(env, policy, V, discount_factor):
    """
    Helper function to evaluate a policy.

    Arguments:
        env: openAI gym env object.
        policy: policy to evaluate.
        V: Estimated Value for each state. Vector of length nS.
        discount_factor: MDP discount factor.
    Return:
        policy_value: Estimated value of each state following a given policy and state-value 'V'.

    """
    policy_value = np.zeros(env.nS)
    for state, action in enumerate(policy):
        for probablity, next_state, reward, info in env.P[state][action]:
            policy_value[state] += probablity * (reward + (discount_factor * V[next_state]))

    return policy_value


def relevant_states_policy_iteration(env, **kwargs):
    gamma = kwargs.get('gamma', 1.0)
    max_iteration = 1000

    # intialize the state-Value function
    V = np.zeros(env.nS)

    # intialize a random policy
    policy_curr = np.random.randint(0, env.nA, env.nS)
    policy_prev = np.copy(policy_curr)

    for i in range(max_iteration):
        # evaluate given policy
        start = time.time()
        V = policy_eval(env, policy_curr, V, gamma)

        # improve policy
        visited_states = set()
        states_to_visit = set([env.s])
        while len(states_to_visit) > 0:
            # import ipdb
            # ipdb.set_trace()
            # print(f'There are {len(states_to_visit)} to visit out of {env.nS}')
            curr_state = states_to_visit.pop()
            visited_states.add(curr_state)
            curr_state_action_values = np.zeros(env.nA)
            # choose the best action and update V
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[curr_state][action]:
                    if reward == env.reward_of_clash and done:
                        curr_state_action_values[action] = -math.inf
                        break

                    curr_state_action_values[action] += prob * (reward + gamma * V[next_state])

            policy_curr[curr_state] = np.argmax(curr_state_action_values)
            for _, next_state, _, _ in env.P[curr_state][policy_prev[curr_state]]:
                if next_state not in visited_states:
                    states_to_visit.add(next_state)

        print(f"visited {len(visited_states)} states out of {env.nS}")
        print(f"iteration {i} over")

        # if policy not changed over 10 iterations it converged.
        if i % 10 == 0:
            if np.all(np.equal(policy_curr, policy_prev)):
                print('better policy converged at iteration %d' % (i + 1))
                break
            policy_prev = np.copy(policy_curr)

        # print(f'better PI: iteration {i + 1} took {time.time() - start} seconds')

    policy = TabularValueFunctionPolicy(env, 1.0)
    policy.v = policy_eval(env, policy_curr, V, gamma)

    return policy


class PolicyIterationPlanner(Planner):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def plan(self, env: MapfEnv,info:Dict, **kwargs) -> Policy:
        gamma = kwargs.get('gamma', 1.0)
        max_iteration = 1000

        # intialize the state-Value function
        V = np.zeros(env.nS)

        # intialize a random policy
        policy_curr = np.random.randint(0, env.nA, env.nS)
        policy_prev = np.copy(policy_curr)

        for i in range(max_iteration):
            # evaluate given policy
            start = time.time()
            V = policy_eval(env, policy_curr, V, gamma)

            # improve policy
            policy_curr = update_policy(env, policy_curr, V, gamma)

            # if policy not changed over 10 iterations it converged.
            if i % 10 == 0:
                if np.all(np.equal(policy_curr, policy_prev)):
                    # print('policy iteration converged at iteration %d' % (i + 1))
                    break
                policy_prev = np.copy(policy_curr)

            # print(f'PI: iteration {i + 1} took {time.time() - start} seconds')

        policy = TabularValueFunctionPolicy(env, 1.0)
        policy.v = policy_eval(env, policy_curr, V, gamma)

        return policy

    def dump_to_str(self):
        return json.dumps({'gamma': self.gamma})

    @staticmethod
    def load_from_str(json_str: str) -> object:
        json_obj = json.loads(json_str)
        return PolicyIterationPlanner(json_obj['gamma'])
