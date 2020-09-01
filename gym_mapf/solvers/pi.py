import time
import numpy as np
import math
from typing import Dict

from gym_mapf.solvers import V_TYPE_SIZE, V_TYPE, MAXIMUM_RAM
from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.solvers.utils import Policy, ValueFunctionPolicy


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
    action_values = np.zeros(env.nA, dtype=V_TYPE)

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
    policy_value = np.zeros(env.nS, dtype=V_TYPE)
    for state, action in enumerate(policy):
        for probablity, next_state, reward, info in env.P[state][action]:
            policy_value[state] += probablity * (reward + (discount_factor * V[next_state]))

    return policy_value


def policy_iteration(gamma: float, env: MapfEnv, info: Dict, **kwargs) -> Policy:
    gamma = kwargs.get('gamma', 1.0)
    max_iteration = 1000

    # intialize the state-Value function
    if V_TYPE_SIZE * env.nS > MAXIMUM_RAM:
        info['end_reason'] == "out_of_memory"
        return None

    V = np.zeros(env.nS, dtype=V_TYPE)

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

    policy = ValueFunctionPolicy(env, 1.0)
    policy.v = policy_eval(env, policy_curr, V, gamma)

    return policy
