from collections import Counter

import numpy as np
from colorama import Fore
from gym import spaces
from gym.envs.toy_text.discrete import DiscreteEnv
from numpy.distutils.system_info import accelerate_info

from gym_mapf.envs import *
from gym_mapf.mapf.grid import EmptyCell, ObstacleCell

CELL_TO_CHAR = {
    EmptyCell: '.',
    ObstacleCell: '@'
}

np_random = np.random.RandomState()

np_random.seed(0)


def stay_if_hit_obstacle(exec_func):
    def new_exec_func(loc, map):
        new_loc = exec_func(loc, map)
        if map[new_loc] is ObstacleCell:
            return loc

        return new_loc

    return new_exec_func


@stay_if_hit_obstacle
def execute_up(loc, map):
    return max(0, loc[0] - 1), loc[1]


@stay_if_hit_obstacle
def execute_down(loc, map):
    return min(len(map) - 1, loc[0] + 1), loc[1]


@stay_if_hit_obstacle
def execute_right(loc, map):
    return loc[0], min(len(map[0]) - 1, loc[1] + 1)


@stay_if_hit_obstacle
def execute_left(loc, map):
    return loc[0], max(0, loc[1] - 1)


def execute_stay(loc, _):
    return loc


ACTION_TO_FUNC = {
    UP: execute_up,
    DOWN: execute_down,
    RIGHT: execute_right,
    LEFT: execute_left,
    STAY: execute_stay
}


def execute_action(grid, s, noised_action):
    new_state = []
    for i, single_action in enumerate(noised_action):
        exec_func = ACTION_TO_FUNC[single_action]
        new_state.append(exec_func(s[i], grid))

    new_state = tuple(new_state)
    return new_state


def vector_state_to_integer(grid, s):
    return vector_to_integer(s, len(grid) * len(grid[0]),
                             lambda v: len(grid[0]) * v[0] + v[1])


def vector_action_to_integer(a):
    return vector_to_integer(a, len(ACTIONS), lambda x: ACTIONS.index(x))


def integer_action_to_vector(a, n_agents):
    return integer_to_vector(a, len(ACTIONS), n_agents, lambda n: ACTIONS[n])


def integer_state_to_vector(s, grid, n_agents):
    return integer_to_vector(s,
                             len(grid) * len(grid[0]),
                             n_agents,
                             lambda n: (int(n / len(grid[0])), n % len(grid[0])))


class StateToActionGetter:
    def __init__(self, grid, n_agents, agents_starts, agents_goals,
                 right_fail, left_fail, reward_of_clash, reward_of_goal, reward_of_living,
                 s):
        self.grid = grid
        self.n_agents = n_agents
        self.agents_starts = agents_starts
        self.agents_goals = agents_goals
        self.s = s
        self.right_fail = right_fail
        self.left_fail = left_fail
        self.reward_of_clash = reward_of_clash
        self.reward_of_goal = reward_of_goal
        self.reward_of_living = reward_of_living

    def get_possible_actions(self, a):
        if len(a) == 1:
            right, left = POSSIBILITIES[a[0]]
            return [
                (self.right_fail, (right,)),
                (self.left_fail, (left,)),
                (1.0 - self.right_fail - self.left_fail, a)
            ]

        head, *tail = a
        tail = tuple(tail)
        right, left = POSSIBILITIES[head]
        res = []
        for prob, noised_action in self.get_possible_actions(tail):
            res += [
                (self.right_fail * prob, (right,) + noised_action),  # The first action noised to right
                (self.left_fail * prob, (left,) + noised_action),  # The first action noised to left
                ((1.0 - self.right_fail - self.left_fail) * prob, (head,) + noised_action)
                # The first action remained the same
            ]

        return res

    def is_terminal(self, s):
        loc_count = Counter(s)
        if len([x for x in loc_count.values() if x > 1]) != 0:  # clash between two agents.
            return True

        goals = [loc == integer_state_to_vector(self.agents_goals, self.grid, self.n_agents)[i]
                 for i, loc in enumerate(s)]
        all_in_goal = all(goals)

        if all_in_goal:
            return True

        return False

    def calc_transition_reward(self, original_state, action, new_state):
        if type(original_state) == int:
            original_state = integer_state_to_vector(original_state, self.grid, self.n_agents)

        if type(new_state) == int:
            new_state = integer_state_to_vector(new_state, self.grid, self.n_agents)

        if type(action) == int:
            action = integer_action_to_vector(action, self.n_agents)

        loc_count = Counter(new_state)
        if len([x for x in loc_count.values() if x > 1]) != 0:  # clash between two agents.
            return self.reward_of_clash, True

        goals = [loc == integer_state_to_vector(self.agents_goals, self.grid, self.n_agents)[i]
                 for i, loc in enumerate(new_state)]
        all_in_goal = all(goals)

        if all_in_goal:
            return self.reward_of_goal, True

        return self.reward_of_living, False

    def __getitem__(self, a):
        s = integer_state_to_vector(self.s, self.grid, self.n_agents)
        a = integer_action_to_vector(a, self.n_agents)

        if self.is_terminal(s):
            return [(1.0, self.s, 0, True)]

        transitions = []
        for i in range(self.n_agents):
            if s[i] == integer_state_to_vector(self.agents_goals, self.grid, self.n_agents)[i]:
                a = a[:i] + (STAY,) + a[(i + 1):]
        for prob, noised_action in self.get_possible_actions(a):
            new_state = execute_action(self.grid, s, noised_action)
            reward, done = self.calc_transition_reward(s, a, new_state)
            similar_transitions = [(p, s, r, d) for (p, s, r, d) in transitions
                                   if s == new_state and r == reward and d == done]
            if len(similar_transitions) > 0:
                idx = transitions.index(similar_transitions[0])
                old_prob, _, _, _ = transitions[idx]
                transitions = transitions[:idx] + [(old_prob + prob, new_state, reward, done)] + transitions[(idx + 1):]
            else:
                transitions.append((prob, new_state, reward, done))

        return [(prob,
                 vector_to_integer(new_state, len(self.grid) * len(self.grid[0]),
                                   lambda v: len(self.grid[0]) * v[0] + v[1]),
                 reward,
                 done)
                for (prob, new_state, reward, done) in transitions]


class StateGetter:
    def __init__(self, grid, n_agents, agents_starts, agent_goals, right_fail, left_fail, reward_of_clash,
                 reward_of_goal,
                 reward_of_living, ):
        self.grid = grid
        self.n_agents = n_agents
        self.agents_starts = agents_starts
        self.agents_goals = agent_goals
        self.right_fail = right_fail
        self.left_fail = left_fail
        self.reward_of_clash = reward_of_clash
        self.reward_of_goal = reward_of_goal
        self.reward_of_living = reward_of_living

    def __getitem__(self, s):
        return StateToActionGetter(self.grid,
                                   self.n_agents,
                                   self.agents_starts,
                                   self.agents_goals,
                                   self.right_fail,
                                   self.left_fail,
                                   self.reward_of_clash,
                                   self.reward_of_goal,
                                   self.reward_of_living,
                                   s)


class MapfEnv(DiscreteEnv):
    # TODO: return to call super c'tor
    def __init__(self, grid, n_agents, agents_starts, agents_goals,
                 right_fail, left_fail, reward_of_clash, reward_of_goal, reward_of_living):

        self.grid = grid
        self.agents_starts, self.agents_goals = agents_starts, agents_goals
        self.n_agents = n_agents
        self.right_fail = right_fail
        self.left_fail = left_fail
        self.reward_of_clash = reward_of_clash
        self.reward_of_goal = reward_of_goal
        self.reward_of_living = reward_of_living

        self.nS = (len(self.grid) * len(self.grid[0])) ** self.n_agents  # each agent may be in each of the cells.
        self.nA = len(ACTIONS) ** self.n_agents
        self.P = StateGetter(self.grid, self.n_agents, self.agents_starts, agents_goals, right_fail, left_fail,
                             reward_of_clash, reward_of_goal, reward_of_living)
        # self.isd = [1.0] + [0.0] * (self.nS - 1)  # irrelevant.
        self.lastaction = None  # for rendering

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        self.makespan = 0
        self.soc = 0

        self.seed()
        self.reset()

    def step(self, a):
        s = integer_state_to_vector(self.s, self.grid, self.n_agents)
        agents_goals = integer_state_to_vector(self.agents_goals, self.grid, self.n_agents)
        n_agents_in_goals = len([i for i in range(self.n_agents) if s[i] == agents_goals[i]])

        self.soc += self.n_agents - n_agents_in_goals
        self.makespan += 1
        return super().step(a)

    def reset(self):
        self.lastaction = None
        self.makespan = 0
        self.soc = 0
        self.s = self.agents_starts
        return self.s

    def render(self, mode='human'):
        # init(autoreset=True)

        for i in range(len(self.grid)):
            print('')  # newline
            for j in range(len(self.grid[0])):
                if (i, j) in self.s and (i, j) in self.agents_goals:
                    # print an agent which reached it's goal
                    print(Fore.GREEN + str(self.s.index((i, j))) + Fore.RESET, end=' ')
                    continue
                if (i, j) in self.s:
                    print(Fore.YELLOW + str(self.s.index((i, j))) + Fore.RESET, end=' ')
                    continue
                if (i, j) in self.agents_goals:
                    print(Fore.BLUE + str(self.agents_goals.index((i, j))) + Fore.RESET, end=' ')
                    continue

                print(CELL_TO_CHAR[self.grid[i, j]], end=' ')
