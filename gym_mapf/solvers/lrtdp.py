import time
from typing import Callable, Dict

from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.solvers.rtdp import RtdpPolicy, greedy_action
from gym_mapf.solvers.utils import Planner, Policy


class LrtdpPolicy(RtdpPolicy):
    def __init__(self, env, gamma, heuristic):
        super().__init__(env, gamma, heuristic)
        self.solved = set()


class LrtdpPlanner(Planner):
    """Labeled RTDP"""

    def __init__(self, heuristic_function: Callable[[MapfEnv], Callable[[int], float]], max_iterations: int,
                 gamma: float, epsilon: float):
        super().__init__()
        self.heuristic_function = heuristic_function
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.epsilon = epsilon

    def dump_to_str(self):
        pass

    @staticmethod
    def load_from_str(json_str: str) -> object:
        pass

    def residual(self, policy, s):
        return abs(policy.v[s] -
                   sum([prob * (reward + self.gamma * policy.v[next_state])
                        for prob, next_state, reward, done in policy.env.P[s][policy.act(s)]]))

    def check_solved(self, policy, s):
        # if s in policy.solved:
        #     return True
        ret = True
        open = [s]
        closed = []
        while open:
            expanded_state = open.pop()
            closed.append(expanded_state)

            if self.residual(policy, expanded_state) > self.epsilon:
                ret = False
                continue

            # expand state
            action = policy.act(expanded_state)
            for prob, next_state, reward, done in policy.env.P[expanded_state][action]:
                if all([next_state not in policy.solved,
                        next_state not in open,
                        next_state not in closed]):
                    open.append(next_state)

        if ret:
            for s in closed:
                policy.solved.add(s)
        else:
            # This is the reverse update
            while closed:
                s = closed.pop()
                a = greedy_action(policy.env, s, policy.v, self.gamma)
                new_v_s = sum([prob * (reward + self.gamma * policy.v[next_state])
                               for prob, next_state, reward, done in policy.env.P[s][a]])
                policy.v_partial_table[s] = new_v_s

        return ret

    def plan(self, env: MapfEnv, info: Dict, **kwargs) -> Policy:
        info['iterations'] = []

        # initialize V to an upper bound
        env.reset()
        initial_state = env.s
        policy = LrtdpPolicy(env, self.gamma, self.heuristic_function(env))

        # follow the greedy policy, for each transition do a bellman update on V
        n_iterations = 0
        while initial_state not in policy.solved and n_iterations < self.max_iterations:
            n_iterations += 1
            s = env.s
            start = time.time()
            path = []

            # LRTDP Trial
            while s not in policy.solved:
                # env.render()
                a = greedy_action(env, s, policy.v, self.gamma)
                path.append((s, a))
                # print(f'action {integer_action_to_vector(a, env.n_agents)} chosen')
                # time.sleep(1)
                new_v_s = sum([prob * (reward + self.gamma * policy.v[next_state])
                               for prob, next_state, reward, done in env.P[s][a]])
                policy.v_partial_table[s] = new_v_s

                # simulate the step and sample a new state
                s, r, done, _ = env.step(a)
                if done:
                    # add the state to done, the action does not matter
                    path.append((s, 0))
                    break

            # iteration finished
            n_moves = len(path)
            while path:
                state, action = path.pop()
                if not self.check_solved(policy, state):
                    break

            info['iterations'].append({
                'n_moves': len(path),
                'time': time.time() - start,
                'n_states_solved': len(policy.solved),
                'final_reward': r,
            })

            env.reset()

        env.reset()

        return policy
