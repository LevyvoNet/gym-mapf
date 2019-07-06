from collections import Counter

import numpy as np
from colorama import Fore
from gym import spaces
from gym.envs.toy_text.discrete import DiscreteEnv

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


class StateToActionGetter:
    def __init__(self, grid, agents_starts, agents_goals,
                 right_fail, left_fail, reward_of_clash, reward_of_goal, reward_of_living,
                 s):
        self.grid = grid
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

    def calc_transition_reward(self, original_state, action, new_state):
        loc_count = Counter(new_state)
        if len([x for x in loc_count.values() if x > 1]) != 0:  # clash between two agents.
            return self.reward_of_clash, True

        goals = [loc == self.agents_goals[i]
                 for i, loc in enumerate(new_state)]
        all_in_goal = all(goals)

        if all_in_goal:
            return self.reward_of_goal, True

        return self.reward_of_living, False

    def __getitem__(self, a):
        if type(a) == int:
            a = integer_to_vector(a, len(ACTIONS), len(self.agents_goals), lambda n: ACTIONS[n])

        transitions = []
        for i in range(len(self.agents_starts)):
            if self.s[i] == self.agents_goals[i]:
                a = a[:i] + (STAY,) + a[(i + 1):]
        for prob, noised_action in self.get_possible_actions(a):
            new_state = execute_action(self.grid, self.s, noised_action)
            reward, done = self.calc_transition_reward(self.s, a, new_state)
            similar_transitions = [(p, s, r, d) for (p, s, r, d) in transitions
                                   if s == new_state and r == reward and d == done]
            if len(similar_transitions) > 0:
                idx = transitions.index(similar_transitions[0])
                old_prob, _, _, _ = transitions[idx]
                transitions = transitions[:idx] + [(old_prob + prob, new_state, reward, done)] + transitions[(idx + 1):]
            else:
                transitions.append((prob, new_state, reward, done))

        return transitions


class StateGetter:
    def __init__(self, grid, agents_starts, agent_goals, right_fail, left_fail, reward_of_clash, reward_of_goal,
                 reward_of_living, ):
        self.grid = grid
        self.agents_starts = agents_starts
        self.agents_goals = agent_goals
        self.right_fail = right_fail
        self.left_fail = left_fail
        self.reward_of_clash = reward_of_clash
        self.reward_of_goal = reward_of_goal
        self.reward_of_living = reward_of_living

    def __getitem__(self, s):
        if type(s) == int:
            s = integer_to_vector(s,
                                  len(self.grid) * len(self.grid[0]),
                                  len(self.agents_starts),
                                  lambda n: (int(n / len(self.grid[0])), n % len(self.grid[0])))

        return StateToActionGetter(self.grid,
                                   self.agents_starts,
                                   self.agents_goals,
                                   self.right_fail,
                                   self.left_fail,
                                   self.reward_of_clash,
                                   self.reward_of_goal,
                                   self.reward_of_living,
                                   s)


class SingleActionSpace(spaces.Discrete):
    NUM_TO_ACTION = {
        0: UP,
        1: RIGHT,
        2: DOWN,
        3: LEFT,
        4: STAY
    }

    def __init__(self):
        super().__init__(5)

    def sample(self):
        return self.NUM_TO_ACTION[np_random.randint(self.n)]

    def __repr__(self):
        return "Discrete(%s, %s, %s, %s, %s)" % (UP, RIGHT, DOWN, LEFT, STAY)


class MapfEnv(DiscreteEnv):

    # TODO: return to call super c'tor
    def __init__(self, grid, agents_starts, agents_goals,
                 right_fail, left_fail, reward_of_clash, reward_of_goal, reward_of_living):
        if len(agents_starts) != len(agents_goals):
            raise Exception("Illegal Arguments - agents starts and goals must have the same length")

        self.grid = grid
        self.agents_starts, self.agents_goals = agents_starts, agents_goals
        self.n_agents = len(agents_starts)
        self.right_fail = right_fail
        self.left_fail = left_fail
        self.reward_of_clash = reward_of_clash
        self.reward_of_goal = reward_of_goal
        self.reward_of_living = reward_of_living

        self.nS = (len(self.grid) * len(self.grid[0])) ** self.n_agents  # each agent may be in each of the cells.
        self.nA = len(ACTIONS) ** self.n_agents
        self.P = StateGetter(self.grid, self.agents_starts, agents_goals, right_fail, left_fail,
                             reward_of_clash, reward_of_goal, reward_of_living)
        self.isd = [1.0] + [0.0] * (self.nS - 1)  # irrelevant.
        self.lastaction = None  # for rendering

        self.action_space = spaces.Tuple([SingleActionSpace()] * self.n_agents)
        self.observation_space = spaces.Tuple(
            [spaces.Tuple(
                [spaces.Discrete(len(self.grid)),
                 spaces.Discrete(len(self.grid[0]))])] * self.n_agents)
        self.makespan = 0
        self.soc = 0

        self.seed()
        self.reset()

    def step(self, a):
        n_agents_in_goals = len([i for i in range(self.n_agents) if self.s[i] == self.agents_goals[i]])

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
