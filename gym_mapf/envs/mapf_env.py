from collections import Counter, defaultdict
from typing import Callable
import itertools
import functools
import enum

from colorama import Fore
from gym import spaces
import gym
from gym.envs.toy_text.discrete import categorical_sample
from gym.utils.seeding import np_random

from gym_mapf.envs.grid import MapfGrid
from gym_mapf.envs import *
from gym_mapf.envs.grid import EmptyCell, ObstacleCell

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


class OptimizationCriteria(enum.Enum):
    SoC = 'SoC'
    Makespan = 'Makespan'


def empty_indices():
    return {'prev': [], 'next': []}


GYM_MAPF_SEED = 42


def stay_if_hit_obstacle(exec_func):
    def new_exec_func(loc, map):
        new_loc = exec_func(loc, map)
        if map[new_loc] is ObstacleCell:
            return loc

        return new_loc

    return new_exec_func


@stay_if_hit_obstacle
def execute_up(loc, _):
    return max(0, loc[0] - 1), loc[1]


@stay_if_hit_obstacle
def execute_down(loc, map):
    return min(len(map) - 1, loc[0] + 1), loc[1]


@stay_if_hit_obstacle
def execute_right(loc, map):
    return loc[0], min(len(map[0]) - 1, loc[1] + 1)


@stay_if_hit_obstacle
def execute_left(loc, _):
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
    return vector_to_integer(a, [len(ACTIONS)] * len(a), lambda x: ACTIONS.index(x))


def integer_action_to_vector(a, n_agents):
    return integer_to_vector(a, [len(ACTIONS)] * n_agents, n_agents, lambda n: ACTIONS[n])


def function_to_get_item_of_object(func):
    """Return an object which its __get_item_ calls the given function"""

    class ret_type:
        def __getitem__(self, item):
            return func(item)

    return ret_type()


class MapfEnv(gym.Env):
    def __init__(self,
                 grid: MapfGrid,
                 n_agents: int,
                 start_locations: tuple,
                 goal_locations: tuple,
                 fail_prob: float,
                 reward_of_collision: float,
                 reward_of_goal: float,
                 reward_of_living: float,
                 optimization_criteria: OptimizationCriteria):
        # Constants
        self.grid = grid
        self.agents_starts, self.agents_goals = start_locations, goal_locations
        self.n_agents = n_agents
        self.fail_prob = fail_prob
        self.right_fail = self.fail_prob / 2
        self.left_fail = self.fail_prob / 2
        self.reward_of_clash = reward_of_collision
        self.reward_of_goal = reward_of_goal
        self.reward_of_living = reward_of_living
        self.optimization_criteria = optimization_criteria

        # Random Parameters
        self.np_random, self.seed = np_random(GYM_MAPF_SEED)

        # Initialize the match between state numbers and locations on grid
        self.valid_locations = [loc for loc in self.grid if self.grid[loc] is EmptyCell]
        self.loc_to_int = {loc: i for i, loc in enumerate(self.valid_locations)}

        self.nS = len(self.valid_locations) ** self.n_agents  # each agent may be in each of the cells.
        self.nA = len(ACTIONS) ** self.n_agents

        # This is an object which its __get_item__ expects s and returns an object which expects a
        self.P = function_to_get_item_of_object(self._partial_get_transitions)
        # self.P = StateGetter(self)

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.state_to_locations_cache = {}
        self.locations_to_state_cache = {}
        self.predecessors_cache = {}
        self._single_location_predecessors_cache = {}
        self.get_possible_actions_cache = {}
        self.single_agent_movements_cache = {}
        self.is_terminal_cache = {}
        self.transitions_cache = {}
        self.s_transitions_cache = {}

        self.reset()

        # This will throw an exception if the goal coordinates are illegal (an obstacle)
        self.locations_to_state(self.agents_goals)

        # State of the env (all of these values shall be reset during the reset method)
        self.lastaction = None  # for rendering
        self.s = None

        # Initialize the match between state numbers and locations on grid
        self.valid_locations = [loc for loc in self.grid if self.grid[loc] is EmptyCell]
        self.loc_to_int = {loc: i for i, loc in enumerate(self.valid_locations)}

        self.nS = len(self.valid_locations) ** self.n_agents  # each agent may be in each of the cells.
        self.nA = len(ACTIONS) ** self.n_agents

        # This is an object which its __get_item__ expects s and returns an object which expects a
        self.P = function_to_get_item_of_object(self._partial_get_transitions)
        # self.P = StateGetter(self)

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.state_to_locations_cache = {}
        self.locations_to_state_cache = {}
        self.predecessors_cache = {}
        self._single_location_predecessors_cache = {}
        self.get_possible_actions_cache = {}
        self.single_agent_movements_cache = {}
        self.is_terminal_cache = {}
        self.transitions_cache = {}
        self.s_transitions_cache = {}

        # Set the env to its start state
        self.reset()

        # This will throw an exception if the goal coordinates are illegal (an obstacle)
        self.locations_to_state(self.agents_goals)

    def single_agent_movements(self, local_state, a):
        ret = self.single_agent_movements_cache.get((local_state, a), None)
        if ret is not None:
            return ret

        location = self.valid_locations[local_state]
        right_a, left_a = POSSIBILITIES[ACTIONS[a]]
        noised_actions_and_probability = [
            (1 - self.right_fail - self.left_fail, ACTIONS[a]),
            (self.right_fail, right_a),
            (self.left_fail, left_a)
        ]
        noised_actions_and_probability = [(p, a) for (p, a) in noised_actions_and_probability if p > 0]
        movements = []
        movements_next_state = []
        for prob, noised_action in noised_actions_and_probability:
            next_state = self.loc_to_int[execute_action(self.grid, (location,), (noised_action,))[0]]
            if next_state in movements_next_state:
                movement_index = movements_next_state.index(next_state)
                movements[movement_index] = (local_state, next_state, movements[movement_index][2] + prob)
            else:
                movements.append((local_state, next_state, prob))
                movements_next_state.append(next_state)

        self.single_agent_movements_cache[(local_state, a)] = movements
        return movements

    def get_possible_actions(self, a):
        ret = self.get_possible_actions_cache.get(a, None)
        if ret is not None:
            return ret

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

        self.get_possible_actions_cache[a] = res
        return res

    def is_terminal(self, s):
        ret = self.is_terminal_cache.get(s, None)
        if ret is not None:
            return ret

        loc_count = Counter(s)
        if len([x for x in loc_count.values() if x > 1]) != 0:  # clash between two agents.
            self.is_terminal_cache[s] = True
            return True

        goals = [loc == self.agents_goals[i]
                 for i, loc in enumerate(s)]
        all_in_goal = all(goals)

        if all_in_goal:
            self.is_terminal_cache[s] = True
            return True

        self.is_terminal_cache[s] = False
        return False

    def calc_transition_reward_from_local_states(self, prev_local_states, action: int, next_local_states):
        living_reward = self._living_reward(prev_local_states, action)

        if self._is_collision_transition_from_local_states(prev_local_states, next_local_states):
            return self.reward_of_clash + living_reward, True

        if all([self.loc_to_int[self.agents_goals[i]] == next_local_states[i] for i in range(self.n_agents)]):
            # goal state
            return self.reward_of_goal + living_reward, True

        return self.reward_of_living, False

    def step(self, a: int):
        state_locations = self.state_to_locations(self.s)
        if self.is_terminal(state_locations):
            return self.s, 0, True, {"prob": 0}

        single_agent_actions = [ACTIONS_TO_INT[single_agent_action]
                                for single_agent_action in integer_action_to_vector(a, self.n_agents)]
        single_agent_states = [self.loc_to_int[single_agent_loc]
                               for single_agent_loc in state_locations]
        single_agent_movements = [self.single_agent_movements(single_agent_states[i], single_agent_actions[i])
                                  for i in range(self.n_agents)]

        next_local_states = []
        total_prob = 1
        # single_agent_movements is a list of [[(s1,s'1,p), (s1,s''1,p),...], [(s2,s'2,p), (s2,s''2,p),...]], ...]
        # We want to first choose the transition for agent1, then for agent2, etc.
        for agent_movements in single_agent_movements:
            probs = [t[2] for t in agent_movements]
            chosen_movement_idx = categorical_sample(probs, self.np_random)
            next_local_states.append(agent_movements[chosen_movement_idx][1])
            total_prob *= agent_movements[chosen_movement_idx][2]

        # next_local_states holds a list of the local states of the agents
        next_locations = tuple([self.valid_locations[local_state] for local_state in next_local_states])
        new_state = self.locations_to_state(next_locations)
        reward, done = self.calc_transition_reward_from_local_states(single_agent_states, a, next_local_states)

        self.s = new_state
        return new_state, reward, done, {"prob": total_prob}

    def get_single_agent_transitions(self, joint_state, agent_idx, a):
        agents_locations = self.state_to_locations(joint_state)
        local_states = [self.loc_to_int[loc] for loc in agents_locations]
        transitions = []
        for (_, next_state, prob) in self.single_agent_movements(local_states[agent_idx], a):
            next_location = self.valid_locations[next_state]
            next_agents_locations = agents_locations[:agent_idx] + (next_location,) + agents_locations[agent_idx + 1:]
            next_joint_state = self.locations_to_state(next_agents_locations)
            reward, done = self.calc_transition_reward_from_local_states(
                local_states,
                None,
                local_states[:agent_idx] + [next_state] + local_states[agent_idx + 1:])
            transitions.append((prob, next_joint_state, reward, done))

        return transitions

    def step_single_agent(self, agent_idx, a):
        transitions = self.get_single_agent_transitions(self.s, agent_idx, a)
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, new_state, r, d = transitions[i]
        return new_state, r, d, {"prob": p}

    def reset(self):
        self.lastaction = None
        self.s = self.locations_to_state(self.agents_starts)
        return self.s

    def render(self, mode='human'):
        # init(autoreset=True)
        v_state = self.state_to_locations(self.s)
        v_agent_goals = self.agents_goals

        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                # print((i, j))
                if (i, j) in v_state:
                    first_index = v_state.index((i, j))
                    if (i, j) in v_state[first_index + 1:]:
                        # There is another agent in this cell, a collision happened
                        print(Fore.RED + '*' + Fore.RESET, end=' ')
                    else:
                        # This is the only agent in this cell
                        if (i, j) in v_agent_goals and v_agent_goals.index((i, j)) == v_state.index((i, j)):
                            # print an agent which reached it's goal
                            print(Fore.GREEN + str(v_state.index((i, j))) + Fore.RESET, end=' ')
                            continue
                        print(Fore.YELLOW + str(first_index) + Fore.RESET, end=' ')
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

    def state_to_locations(self, state):
        if state in self.state_to_locations_cache:
            return self.state_to_locations_cache[state]

        ret = integer_to_vector(state, [len(self.valid_locations)] * self.n_agents, self.n_agents,
                                lambda x: self.valid_locations[x])
        self.state_to_locations_cache[state] = ret
        return ret

    def locations_to_state(self, locs):
        if locs in self.locations_to_state_cache:
            return self.locations_to_state_cache[locs]

        if self.n_agents != len(locs):
            raise AssertionError(
                f'{locs} locations number is different than the number of agents {self.n_agents}')

        local_state_vector = tuple([self.loc_to_int[loc] for loc in locs])
        ret = vector_to_integer(local_state_vector, [len(self.valid_locations)] * len(local_state_vector), lambda x: x)
        self.locations_to_state_cache[locs] = ret
        return ret

    def predecessors(self, s: int):
        ret = self.predecessors_cache.get(s, None)
        if ret is not None:
            return ret

        ret = set([self.locations_to_state(loc_vec) for loc_vec in
                   self._multiple_locations_predecessors(self.state_to_locations(s))])

        self.predecessors_cache[s] = ret
        return ret

    def _is_collision_transition_from_local_states(self, prev_local_states, next_local_states):
        states_data = defaultdict(empty_indices)
        for i, (prev_state, next_state) in enumerate(zip(prev_local_states, next_local_states)):
            states_data[prev_state]['prev'].append(i)
            states_data[next_state]['next'].append(i)

            if len(states_data[next_state]['next']) > 1:
                # collision happened
                return True

            if len(states_data[next_state]['next']) > 0 and len(states_data[next_state]['prev']) > 0:
                # there is an agent in next_state and there was before as well, find out if it the same one
                next_agent = states_data[next_state]['next'][0]
                prev_agent = states_data[next_state]['prev'][0]
                if next_agent != prev_agent:
                    # It is not the same, check out if the new agent switched with the old one
                    if prev_local_states[next_agent] == next_local_states[prev_agent]:
                        # switch between agents is also a clash
                        return True

        return False

    @functools.lru_cache(maxsize=None)
    def is_collision_transition(self, s: int, next_state: int):
        prev_locations = self.state_to_locations(s)
        prev_local_states = [self.loc_to_int[single_agent_loc]
                             for single_agent_loc in prev_locations]

        next_locations = self.state_to_locations(next_state)
        next_local_states = [self.loc_to_int[single_agent_loc]
                             for single_agent_loc in next_locations]

        return self._is_collision_transition_from_local_states(prev_local_states, next_local_states)

    # Private Methods

    def _single_location_predecessors(self, loc):
        ret = self._single_location_predecessors_cache.get(loc, None)
        if ret is not None:
            return ret

        by_up = execute_action(self.grid, loc, (DOWN,))
        by_down = execute_action(self.grid, loc, (UP,))
        by_right = execute_action(self.grid, loc, (LEFT,))
        by_left = execute_action(self.grid, loc, (RIGHT,))
        by_stay = execute_action(self.grid, loc, (STAY,))

        ret = [loc for loc in [by_up, by_down, by_right, by_left, by_stay]
               if self.grid[loc[0]] is EmptyCell]

        self._single_location_predecessors_cache[loc] = ret
        return ret

    def _multiple_locations_predecessors(self, locs):
        head = self._single_location_predecessors((locs[0],))
        if len(locs) == 1:
            return head

        return [first_loc + partial_location
                for partial_location in self._multiple_locations_predecessors(locs[1:])
                for first_loc in head]

    def _living_reward(self, prev_locations: tuple, a: int):
        if self.optimization_criteria == OptimizationCriteria.Makespan:
            return self.reward_of_living

        # SoC - an agent "pays" REWARD_OF_LIVING unless it reached its goal state and stayed there
        vector_a = integer_action_to_vector(a, self.n_agents)
        n_agents_stayed_in_goals = len([i for i in range(self.n_agents)
                                        if prev_locations[i] == self.agents_goals[i] and vector_a[i] == STAY])

        return (self.n_agents - n_agents_stayed_in_goals) * self.reward_of_living

    # TODO: consider return as part of the transition whether there has been a collision or not for efficiency. At the
    #   current the solver calls is_collision_transition which opens up the locations for each transition
    def _get_transitions(self, s, a):
        """Return transitions given a state and an action from that state

        The transitions are in form of (prob, new_state, reward, done)
        """
        s_a_cache = self.transitions_cache.get(s, None)
        if s_a_cache is not None:
            ret = s_a_cache.get(a, None)
            if ret is not None:
                return ret

        state_locations = self.state_to_locations(s)
        if self.is_terminal(state_locations):
            return [(1.0, s, 0, True)]

        single_agent_actions = [ACTIONS_TO_INT[single_agent_action]
                                for single_agent_action in integer_action_to_vector(a, self.n_agents)]
        single_agent_states = [self.loc_to_int[single_agent_loc]
                               for single_agent_loc in state_locations]
        single_agent_movements = [self.single_agent_movements(single_agent_states[i], single_agent_actions[i])
                                  for i in range(self.n_agents)]

        transitions = []
        # TODO: check single_agent_movements when the action is all stay
        for comb in itertools.product(*single_agent_movements):
            prob = functools.reduce(lambda x, y: x * y, [x[2] for x in comb])
            multiagent_next_locations = tuple([self.valid_locations[int(s)] for s in
                                               [x[1] for x in comb]])

            multiagent_next_state = self.locations_to_state(multiagent_next_locations)

            reward, done = self.calc_transition_reward_from_local_states([x[0] for x in comb], a,
                                                                         [x[1] for x in comb])

            transitions.append((prob, multiagent_next_state, reward, done))

        if s_a_cache is None:
            self.transitions_cache[s] = {}
        self.transitions_cache[s][a] = transitions

        return transitions

    def _partial_get_transitions(self, s):
        ret = self.s_transitions_cache.get(s, None)
        if ret is not None:
            return ret

        ret = function_to_get_item_of_object(functools.partial(self._get_transitions, s))
        self.s_transitions_cache[s] = ret
        return ret
