import gym
from gym import spaces
from gym.envs.toy_text.discrete import DiscreteEnv
from gym_mapf.utils.executor import (UP,
                                     DOWN,
                                     RIGHT,
                                     LEFT,
                                     STAY,
                                     ACTIONS,
                                     execute_action)
from gym_mapf.utils.state import MapfState
from collections import Counter
from gym_mapf.utils.grid import EmptyCell, ObstacleCell
from colorama import Fore

CELL_TO_CHAR = {
    EmptyCell: '.',
    ObstacleCell: ''
}

POSSIBILITIES = {
    UP: (RIGHT, LEFT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (DOWN, UP),
    STAY: (STAY, STAY)
}


class StateToActionGetter:
    def __init__(self, grid, agent_starts, agent_goals, right_fail, left_fail, s):
        self.map = grid
        self.agent_starts = agent_starts
        self.agent_goals = agent_goals
        self.s = s
        self.right_fail = right_fail
        self.left_fail = left_fail

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
        loc_count = Counter(new_state.agent_locations)
        if len([x for x in loc_count.values() if x > 1]) != 0:  # clash between two agents.
            return -1000.0, True

        goals = [loc == self.agent_goals[i]
                 for i, loc in enumerate(new_state.agent_locations)]
        all_in_goal = all(goals)

        if all_in_goal:
            return 100.0 * len(goals), True

        return -5.0, False

    def __getitem__(self, a):
        transitions = []
        # TODO: sum the probabilities for equal states?
        for prob, noised_action in self.get_possible_actions(a):
            new_state = execute_action(self.s, noised_action)
            reward, done = self.calc_transition_reward(self.s, a, new_state)
            transitions.append((prob, new_state, reward, done))

        return transitions


class StateGetter:
    def __init__(self, grid, agents_starts, agent_goals, right_fail, left_fail):
        self.grid = grid
        self.agents_starts = agents_starts
        self.agents_goals = agent_goals
        self.right_fail = right_fail
        self.left_fail = left_fail

    def __getitem__(self, s):
        return StateToActionGetter(self.grid,
                                   self.agents_starts,
                                   self.agents_goals,
                                   self.right_fail,
                                   self.left_fail,
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
        return self.NUM_TO_ACTION[gym.spaces.np_random.randint(self.n)]

    def __repr__(self):
        return "Discrete(%s, %s, %s, %s, %s)" % (UP, RIGHT, DOWN, LEFT, STAY)


# TODO: when one of the agents reaches the goal should it stop moving?
class MapfEnv(DiscreteEnv):
    class StateGetter:
        def __getitem__(self, s):
            return

    def __init__(self, grid, agents_starts, agents_goals):
        if len(agents_starts) != len(agents_goals):
            raise Exception("Illegal Arguments - agents starts and goals must have the same length")

        right_fail = 0.1
        left_fail = 0.1

        self.grid = grid
        self.agents_starts, self.agents_goals = agents_starts, agents_goals
        n_agents = len(agents_starts)

        self.nS = len(self.grid) * len(self.grid[0]) * n_agents  # each agent may be in each of the cells.
        self.nA = n_agents ** len(ACTIONS)
        self.P = StateGetter(self.grid, self.agents_starts, agents_goals, right_fail, left_fail)
        self.isd = [1.0] + [0.0] * (self.nS - 1)  # irrelevant.
        self.lastaction = None  # for rendering

        self.action_space = spaces.Tuple([SingleActionSpace()] * n_agents)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.reset()

    def reset(self):
        self.lastaction = None
        self.s = MapfState(self.grid, self.agents_starts)
        return self.s

    def render(self, mode='human'):
        # init(autoreset=True)

        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if (i, j) in self.s.agent_locations:
                    char = str(self.s.agent_locations.index((i, j)))
                else:
                    char = CELL_TO_CHAR[self.grid[i, j]]

                if (i, j) in self.agents_goals:
                    print(Fore.GREEN + char + Fore.RESET, end='')
                else:
                    print(char, end='')

            print('')  # newline
