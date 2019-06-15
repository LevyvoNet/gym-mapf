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
from colorama import Fore, init

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
            return -1.0, True

        all_in_goal = all([loc == self.agent_goals[i]
                           for i, loc in enumerate(new_state.agent_locations)])

        if all_in_goal:
            return 1.0, True

        return 0.0, False

    def __getitem__(self, a):
        transitions = []
        # TODO: sum the probabilities for equal states
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


# TODO: when one of the agents reaches the goal it should stop moving.
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

        nS = len(self.grid) * len(self.grid[0]) * n_agents  # each agent may be in each of the cells.
        nA = n_agents ** len(ACTIONS)
        P = StateGetter(self.grid, self.agents_starts, agents_goals, right_fail, left_fail)
        isd = [1.0] + [0.0] * (nS - 1)  # irrelevant.

        super().__init__(nS, nA, P, isd)

    def reset(self):
        self.lastaction = None
        self.s = MapfState(self.grid, self.agents_starts)
        return self.s

    def render(self, mode='human'):
        init(autoreset=True)

        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if (i, j) in self.s.agent_locations:
                    char = str(self.s.agent_locations.index((i, j)))
                else:
                    char = CELL_TO_CHAR[self.grid[i, j]]

                if (i, j) in self.agents_goals:
                    print(Fore.GREEN + char, end='')
                else:
                    print(char, end='')

            print('')  # newline

# class BerlinEnvImp2(gym.envs.Env):
#     """
#     In this implementation we wont calculate all of the different transition in advance.
#     We will perform lazy evaluation instead.
#     """
#     metadata = {'render.modes': ['human']}
#
#     def __init__(self):
#         map_file = '../maps/Berlin_1_256/Berlin_1_256.map'
#         scen_file = '../maps/Berlin_1_256/Berlin_1_256-even-1.scen'
#         n_agents = 10
#         self.right_fail = 0.1
#         self.left_fail = 0.1
#         self.observation_space = 0
#         self.action_space = gym.spaces.Discrete(4)
#         self.map = parse_map_file(map_file)
#         self.agent_locations = parse_scen_file(scen_file, n_agents)
#
#     def noise_single_action(self, action):
#         possibilities = POSSIBILITIES[action]
#         return np.random.choice(possibilities + (action,),
#                                 p=[self.right_fail, self.left_fail, 1.0 - self.right_fail - self.left_fail])
#
#     def noise_joint_action(self, joint_action):
#         """Noise action according to MDP params
#
#         Args:
#             joint_action (iterable): the joint actions for all agents
#
#         Returns:
#             iterable. The new action after skewing the steps according to the MDP params.
#             For example (UP,UP,LEFT,DOWN) has self.right_fail*self.left_fail in order to become (RIGHT, LEFT, lEFT, DOWN).
#         """
#         return tuple(
#             self.noise_single_action(a) for a in joint_action
#         )
#
#     def step(self, action):
#         noised_joint_action = self.noise_joint_action(action)
#
#     def reset(self):
#         pass
#
#     def render(self, mode='human'):
#         pass
#
#     def close(self):
#         pass
