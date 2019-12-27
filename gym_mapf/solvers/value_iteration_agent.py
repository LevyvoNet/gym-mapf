import numpy as np
import stopit

from gym_mapf.solvers.utils import safe_actions


def run_episode(env, policy, gamma=1.0, render=False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma=1.0, n=100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
        run_episode(env, policy, gamma=gamma, render=False)
        for _ in range(n)]
    return np.mean(scores)


def extract_policy(v, env, gamma=1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        possible_actions_from_state = safe_actions(env, s)
        q_sa = np.zeros(len(possible_actions_from_state))
        for a_idx in range(len(possible_actions_from_state)):
            a = possible_actions_from_state[a_idx]
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probabili
                # ty, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a_idx] += (p * (r + gamma * v[s_]))
        policy[s] = possible_actions_from_state[np.argmax(q_sa)]
    return policy


def value_iteration(env, max_time, gamma=1.0):
    """ Value-iteration algorithm"""
    # v=  np.full((env.nS), -1)
    v = np.zeros(env.nS)  # initialize value-function
    max_iterations = 100000
    eps = 1e-2
    # with stopit.SignalTimeout(max_time, swallow_exc=False) as timeout_ctx:
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p * (r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in safe_actions(env, s)]
            v[s] = max(q_sa)

        # debug print
        # if i % 100 == 0:
        #     print(v)

        if (np.sum(np.fabs(prev_v - v)) <= eps):
            # debug print
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break

    # OK, let's check what happened
    # if timeout_ctx.state == timeout_ctx.EXECUTED:
    return v, True

    return v, False


class ValueIterationAgent:
    def train(self, env, **kwargs):
        self.gamma = kwargs.get('gamma', 1.0)
        try:
            self.optimal_v, self.train_converged = value_iteration(env, **kwargs)
        except stopit.utils.TimeoutException as e:
            self.optimal_v, self.train_converged = None, False
            return

        self.policy = extract_policy(self.optimal_v, env, self.gamma)

    def select_best_action(self, env, **kwargs):
        return int(self.policy[env.s])

    def __repr__(self):
        return 'ValueIterationAgent()'


def plan_with_value_iteration(env):
    """Get optimal policy derived from value iteration and its expected reward"""
    vi_agent = ValueIterationAgent()
    vi_agent.train(env, max_time=60 * 5)

    def policy_int_output(s):
        return int(vi_agent.policy[s])

    return vi_agent.optimal_v[env.s], policy_int_output
