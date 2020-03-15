from collections import Counter
from typing import Callable

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

ACTION_TO_CHAR = {
    UP: '^',
    RIGHT: '>',
    DOWN: 'V',
    LEFT: '<',
    STAY: 'S'
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


def vector_action_to_integer(a):
    return vector_to_integer(a, len(ACTIONS), lambda x: ACTIONS.index(x))


def integer_action_to_vector(a, n_agents):
    return integer_to_vector(a, len(ACTIONS), n_agents, lambda n: ACTIONS[n])


class StateToActionGetter:
    def __init__(self, env, s):
        self.env = env
        self.s = s
        self.cache = {}

    def get_possible_actions(self, a):
        if len(a) == 1:
            right, left = POSSIBILITIES[a[0]]
            return [
                (self.env.right_fail, (right,)),
                (self.env.left_fail, (left,)),
                (1.0 - self.env.right_fail - self.env.left_fail, a)
            ]

        head, *tail = a
        tail = tuple(tail)
        right, left = POSSIBILITIES[head]
        res = []
        for prob, noised_action in self.get_possible_actions(tail):
            res += [
                (self.env.right_fail * prob, (right,) + noised_action),  # The first action noised to right
                (self.env.left_fail * prob, (left,) + noised_action),  # The first action noised to left
                ((1.0 - self.env.right_fail - self.env.left_fail) * prob, (head,) + noised_action)
                # The first action remained the same
            ]

        return res

    def is_terminal(self, s):
        loc_count = Counter(s)
        if len([x for x in loc_count.values() if x > 1]) != 0:  # clash between two agents.
            return True

        goals = [loc == self.env.agents_goals[i]
                 for i, loc in enumerate(s)]
        all_in_goal = all(goals)

        if all_in_goal:
            return True

        return False

    def collision_happened(self, old_state, new_state):
        # two agents at the same spot
        loc_count = Counter(new_state)
        if len([x for x in loc_count.values() if x > 1]) != 0:
            return True

        # agents switched spots
        for i in range(self.env.n_agents):
            for j in range(self.env.n_agents):
                if old_state[i] == new_state[j] and old_state[j] == new_state[i] and i != j:
                    return True

        return False

    def calc_transition_reward(self, original_state, action, new_state):
        if type(original_state) == int:
            original_state = self.env.state_to_locations(original_state)

        if type(new_state) == int:
            new_state = self.env.state_to_locations(new_state)

        if type(action) == int:
            action = integer_action_to_vector(action, self.env.n_agents)

        if self.collision_happened(original_state, new_state):
            return self.env.reward_of_clash, True

        goals = [loc == self.env.agents_goals[i]
                 for i, loc in enumerate(new_state)]
        all_in_goal = all(goals)

        if all_in_goal:
            return self.env.reward_of_goal, True

        return self.env.reward_of_living, False

    def __getitem__(self, a):
        ret = self.cache.get(a, None)
        if ret is not None:
            return ret

        curr_location = self.env.state_to_locations(self.s)
        vector_a = integer_action_to_vector(a, self.env.n_agents)

        if self.is_terminal(curr_location):
            return [(1.0, self.s, 0, True)]

        transitions = []
        # for i in range(self.env.n_agents):
        #     if s[i] == self.env.agents_goals[i]:
        #         a = a[:i] + (STAY,) + a[(i + 1):]
        for prob, noised_action in self.get_possible_actions(vector_a):
            new_state = execute_action(self.env.grid, curr_location, noised_action)
            reward, done = self.calc_transition_reward(curr_location, vector_a, new_state)
            similar_transitions = [(p, s, r, d) for (p, s, r, d) in transitions
                                   if s == new_state and r == reward and d == done]
            if len(similar_transitions) > 0:
                idx = transitions.index(similar_transitions[0])
                old_prob, _, _, _ = transitions[idx]
                transitions = transitions[:idx] + [(old_prob + prob, new_state, reward, done)] + transitions[(idx + 1):]
            else:
                transitions.append((prob, new_state, reward, done))

        ret = [(prob,
                self.env.locations_to_state(new_state),
                reward,
                done)
               for (prob, new_state, reward, done) in transitions if prob != 0]
        self.cache[a] = ret
        return ret


class StateGetter:
    def __init__(self, env):
        self.env = env
        self.cache = {}

    def __getitem__(self, s):
        if s in self.cache:
            return self.cache[s]

        if s in self.env.mask:
            return self.env.mask[s]

        ret = StateToActionGetter(self.env, s)
        self.cache[s] = ret
        return ret


class MapfEnv(DiscreteEnv):
    # TODO: return to call super c'tor
    def __init__(self, grid, n_agents, agents_starts, agents_goals,
                 right_fail, left_fail, reward_of_clash, reward_of_goal, reward_of_living,
                 mask=()):

        self.grid = grid
        self.agents_starts, self.agents_goals = agents_starts, agents_goals
        self.n_agents = n_agents
        self.right_fail = right_fail
        self.left_fail = left_fail
        self.reward_of_clash = reward_of_clash
        self.reward_of_goal = reward_of_goal
        self.reward_of_living = reward_of_living

        # Initialize the match between state numbers and locations on grid
        self.valid_locations = [loc for loc in self.grid if self.grid[loc] is EmptyCell]
        self.loc_to_int = {loc: i for i, loc in enumerate(self.valid_locations)}

        self.nS = len(self.valid_locations) ** self.n_agents  # each agent may be in each of the cells.
        self.nA = len(ACTIONS) ** self.n_agents

        # self.isd = [1.0] + [0.0] * (self.nS - 1)  # irrelevant.
        self.lastaction = None  # for rendering

        # take care of masks and special states
        self.mask = mask
        self.nS += len([s for s in self.mask if s > (self.nS - 1)])  # add special states to the state count

        self.P = StateGetter(self)

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        self.makespan = 0
        self.soc = 0

        self.state_to_locations_cache = {}
        self.locations_to_state_cache = {}
        self.predecessors_cache = {}
        self._single_location_predecessors_cache={}

        self.seed()
        self.reset()

    def step(self, a):
        curr_location = self.state_to_locations(self.s)
        vector_a = integer_action_to_vector(a, self.n_agents)
        n_agents_stayed_in_goals = len([i for i in range(self.n_agents)
                                        if curr_location[i] == self.agents_goals[i] and vector_a[i] == STAY])

        self.soc += self.n_agents - n_agents_stayed_in_goals
        self.makespan += 1

        # Perform the step
        new_state, reward, done, info = super().step(a)

        # Update terminated
        self.is_terminated = done

        return new_state, reward, done, info

    def reset(self):
        self.lastaction = None
        self.makespan = 0
        self.soc = 0
        self.s = self.locations_to_state(self.agents_starts)
        self.is_terminated = False
        return self.s

    def render(self, mode='human'):
        # init(autoreset=True)
        # import ipdb
        # ipdb.set_trace()
        v_state = self.state_to_locations(self.s)
        v_agent_goals = self.agents_goals

        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                # print((i, j))
                if (i, j) in v_state and (i, j) in v_agent_goals \
                        and v_agent_goals.index((i, j)) == v_state.index((i, j)):
                    # print an agent which reached it's goal
                    print(Fore.GREEN + str(v_state.index((i, j))) + Fore.RESET, end=' ')
                    continue
                if (i, j) in v_state:
                    print(Fore.YELLOW + str(v_state.index((i, j))) + Fore.RESET, end=' ')
                    continue
                if (i, j) in v_agent_goals:
                    print(Fore.BLUE + str(v_agent_goals.index((i, j))) + Fore.RESET, end=' ')
                    continue

                print(CELL_TO_CHAR[self.grid[i, j]], end=' ')

            print('')  # newline

    def render_with_policy(self, agent: int, policy: Callable[[int], int]):
        print('')

        v_state = self.state_to_locations(self.s)
        v_agents_goals = self.agents_goals

        v_agent_loc = v_state[agent]
        v_agent_goal = v_agents_goals[agent]

        for i in range(len(self.grid)):
            print('')  # newline
            for j in range(len(self.grid[0])):
                if (i, j) == v_agent_goal and v_agent_goal == v_agent_loc:
                    # print an agent which reached it's goal
                    print(Fore.GREEN + str(v_state.index((i, j))) + Fore.RESET, end=' ')
                    continue
                if (i, j) == v_agent_loc:
                    print(Fore.YELLOW + str(v_state.index((i, j))) + Fore.RESET, end=' ')
                    continue
                if (i, j) == v_agent_goal:
                    print(Fore.BLUE + str(v_agents_goals.index((i, j))) + Fore.RESET, end=' ')
                    continue

                vector_current_state = v_state[:agent] + ((i, j),) + v_state[agent + 1:]
                self.locations_to_state(vector_current_state)
                integer_current_state = self.locations_to_state(vector_current_state)
                integer_joint_action = policy(integer_current_state)
                vector_joint_action = integer_action_to_vector(integer_joint_action, self.n_agents)
                agent_action = vector_joint_action[agent]

                print(ACTION_TO_CHAR[agent_action], end=' ')

        print('')

    def set_mask(self, mask):
        self.mask = mask
        self.nS += len([s for s in self.mask if s > (self.nS - 1)])  # add special states to the state count
        self.observation_space = spaces.Discrete(self.nS)
        # self.P.mask = mask

        # remove from cache
        for s in self.mask:
            del self.P.cache[s]

        self.reset()

    def state_to_locations(self, state):
        if state in self.state_to_locations_cache:
            return self.state_to_locations_cache[state]

        ret = integer_to_vector(state, len(self.valid_locations), self.n_agents, lambda x: self.valid_locations[x])
        self.state_to_locations_cache[state] = ret
        return ret

    def locations_to_state(self, locs):
        if locs in self.locations_to_state_cache:
            return self.locations_to_state_cache[locs]

        if self.n_agents != len(locs):
            raise AssertionError(
                '{} locations number is different than the number of agents {}'.format(locs, self.n_agents))

        local_state_vector = tuple([self.loc_to_int[loc] for loc in locs])
        ret = vector_to_integer(local_state_vector, len(self.valid_locations), lambda x: x)
        self.locations_to_state_cache[locs] = ret
        return ret

    def _single_location_predecessors(self, loc):
        ret = self._single_location_predecessors_cache.get(loc, None)
        if ret is not None:
            return ret

        by_up = execute_action(self.grid, loc, (DOWN,))
        by_down = execute_action(self.grid, loc, (UP,))
        by_right = execute_action(self.grid, loc, (LEFT,))
        by_left = execute_action(self.grid, loc, (RIGHT,))
        by_stay = execute_action(self.grid, loc, (STAY,))

        ret = [by_up, by_down, by_right, by_left, by_stay]
        self._single_location_predecessors_cache[loc]=ret
        return ret

    def _multiple_locations_predecessors(self, locs):
        head = self._single_location_predecessors((locs[0],))
        if len(locs) == 1:
            return head

        return [first_loc + partial_location
                for partial_location in self._multiple_locations_predecessors(locs[1:])
                for first_loc in head]

    def predecessors(self, s: int):
        ret = self.predecessors_cache.get(s, None)
        if ret is not None:
            return ret

        ret = set([self.locations_to_state(loc_vec) for loc_vec in
                   self._multiple_locations_predecessors(self.state_to_locations(s))])

        self.predecessors_cache[s] = ret
        return ret
