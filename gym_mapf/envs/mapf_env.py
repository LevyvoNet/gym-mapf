import itertools
import random
from typing import Dict, List

import gym
import numpy as np
import enum
import functools
from colorama import Fore
from gym.envs.toy_text.discrete import categorical_sample
from gym.utils.seeding import np_random

from gym_mapf.envs.grid import (EmptyCell,
                                ObstacleCell,
                                MapfGrid,
                                SingleAgentState,
                                SingleAgentAction,
                                CELL_TO_CHAR)

GYM_MAPF_SEED = 42

NOISED_ACTIONS = {
    # STAY is a certain action
    SingleAgentAction.STAY: (SingleAgentAction.STAY, SingleAgentAction.STAY, SingleAgentAction.STAY),

    SingleAgentAction.UP: (SingleAgentAction.RIGHT, SingleAgentAction.LEFT, SingleAgentAction.UP),
    SingleAgentAction.DOWN: (SingleAgentAction.LEFT, SingleAgentAction.RIGHT, SingleAgentAction.DOWN),
    SingleAgentAction.LEFT: (SingleAgentAction.UP, SingleAgentAction.DOWN, SingleAgentAction.LEFT),
    SingleAgentAction.RIGHT: (SingleAgentAction.DOWN, SingleAgentAction.UP, SingleAgentAction.RIGHT),
}


def function_to_get_item_of_object(func):
    """Return an object which its __get_item_ calls the given function"""

    class ret_type:
        def __getitem__(self, item):
            return func(item)

    return ret_type()


class MultiAgentAction:
    def __init__(self, agent_to_action: Dict[int, SingleAgentAction]):
        self.agent_to_action = agent_to_action

    def __getitem__(self, agent):
        return self.agent_to_action[agent]

    def __setitem__(self, key, value):
        self.agent_to_action[key] = value

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.agent_to_action)})'

    def __eq__(self, other):
        return self.agent_to_action == other.agent_to_action

    def __hash__(self):
        return hash(tuple([a.value for a in self.agent_to_action.values()]))


class MultiAgentState:
    """A mapping between an agent and its state"""

    def __init__(self, states: Dict[int, SingleAgentState], grid):
        self.agent_to_state = states
        self.grid = grid
        self.t = tuple([self.agent_to_state[agent]
                        for agent in sorted(self.agent_to_state.keys())])
        self.hash_value = grid.locations_to_int(self.t)

    def __contains__(self, item: SingleAgentState):
        return item in self.agent_to_state.values()

    def __getitem__(self, agent):
        return self.agent_to_state[agent]

    def who_is_here(self, state: SingleAgentState):
        """Return the agents which are in the given state"""
        return [agent for agent in self.agent_to_state
                if self.agent_to_state[agent] == state]

    def __iter__(self):
        return iter(self.agent_to_state.items())

    def agents(self):
        return self.agent_to_state.keys()

    def __eq__(self, other):
        return self.hash_value == other.hash_value

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.agent_to_state)})'

    def __hash__(self):
        return self.hash_value


class MultiAgentStateSpace(gym.Space):
    def __init__(self, agents: List, grid):
        super().__init__()
        self.agents = agents
        self.n_agents = len(self.agents)
        self.grid = grid

    def __iter__(self):
        return map(lambda locations: MultiAgentState({self.agents[i]: SingleAgentState(locations[i][0], locations[i][1])
                                                      for i in range(self.n_agents)}, self.grid),
                   itertools.product(self.grid.valid_locations, repeat=self.n_agents))


class MultiAgentActionSpace(gym.Space):
    def __init__(self, agents: List):
        super().__init__()
        self.agents = agents
        self.n_agents = len(self.agents)

    def __iter__(self):
        return map(lambda actions: MultiAgentAction({self.agents[i]: actions[i] for i in range(self.n_agents)}),
                   itertools.product(SingleAgentAction, repeat=self.n_agents))

    def sample(self):
        return random.choice(list(iter(self)))


class OptimizationCriteria(enum.Enum):
    SoC = 'SoC'
    Makespan = 'Makespan'


class MapfEnv(gym.Env):
    metadata = {'render.modes': ['characters']}

    def __init__(self,
                 grid: MapfGrid,
                 n_agents: int,
                 start_state: MultiAgentState,
                 goal_state: MultiAgentState,
                 fail_prob: float,
                 reward_of_collision: float,
                 reward_of_goal: float,
                 reward_of_living: float,
                 optimization_criteria: OptimizationCriteria):
        if set(start_state.agent_to_state.keys()) != set(goal_state.agent_to_state.keys()):
            raise ValueError(f"agents of start state and goal state are not the same\n"
                             f"start state agents: {sorted(start_state.agent_to_state.keys())}\n"
                             f"goal state agents: {sorted(goal_state.agent_to_state.keys())}\n")
        self.agents = sorted(start_state.agent_to_state.keys())

        # Seed random generators
        self.np_random, self.seed = np_random(GYM_MAPF_SEED)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Save parameters
        self.grid = grid
        self.n_agents = n_agents
        self.optimization_criteria = optimization_criteria

        # Reward calculation parameters
        self.start_state = start_state
        self.goal_state = goal_state
        self.fail_prob = fail_prob
        self.reward_of_collision = reward_of_collision
        self.reward_of_goal = reward_of_goal
        self.reward_of_living = reward_of_living

        # Initialize action and observation spaces
        self.action_space = MultiAgentActionSpace(self.agents)
        self.observation_space = MultiAgentStateSpace(self.agents, self.grid)
        self.P = function_to_get_item_of_object(self._get_transitions_curried)

        # TODO: Is these fields really required?
        self.nS = len(self.grid.valid_locations) ** self.n_agents  # each agent may be in each of the cells.
        self.nA = len(SingleAgentAction) ** self.n_agents

        # Initialize the state of the env
        self.s = self.start_state
        self.terminated = False

    def step(self, action: MultiAgentAction):
        # If the env is terminated, do nothing
        if self.terminated:
            return self.s, 0, True, {"prob": 1}

        # Sample the next state and its probability
        agent_to_next_state = {}
        total_prob = 1
        for agent, state in self.s:
            next_state, prob = self._sample_next_state(state, action[agent])
            total_prob *= prob
            agent_to_next_state[agent] = next_state
        next_state = MultiAgentState(agent_to_next_state, self.grid)

        # Calculate the reward for the transition
        reward, done = self._reward_done_calculation(self.s, next_state)
        if done:
            self.terminated = True

        self.s = next_state
        return next_state, reward, done, {"prob": total_prob}

    def reset(self):
        self.s = self.start_state
        self.terminated = False
        return self.s

    def render(self, mode='characters'):
        # v_state = self.state_to_locations(self.s)
        # v_agent_goals = self.agents_goals
        for i in range(len(self.grid.map)):
            for j in range(len(self.grid.map[0])):
                curr_state = SingleAgentState(i, j)
                if curr_state in self.s:
                    agents_in_state = self.s.who_is_here(curr_state)
                    assert len(agents_in_state) >= 1
                    if len(agents_in_state) > 1:
                        # There is another agent in this cell, a collision happened
                        print(Fore.RED + '*' + Fore.RESET, end=' ')
                    else:
                        # This is the only agent in this cell
                        if all([
                            curr_state in self.goal_state,
                            agents_in_state[0] in self.goal_state.who_is_here(curr_state)
                        ]):
                            # print an agent which reached it's goal
                            print(Fore.GREEN + str(agents_in_state[0]) + Fore.RESET, end=' ')
                            continue
                        print(Fore.YELLOW + str(agents_in_state[0]) + Fore.RESET, end=' ')
                    continue
                if curr_state in self.goal_state:
                    print(Fore.BLUE + str(self.goal_state.who_is_here(curr_state)[0]) + Fore.RESET, end=' ')
                    continue

                print(CELL_TO_CHAR[self.grid[curr_state]], end=' ')

            print('')  # newline

    def close(self):
        super(self.__class__, self).close()

    def seed(self, seed=None):
        np.random.seed(self.seed)

    # Custom methods for MapfEnv only
    @functools.lru_cache(maxsize=None)
    def predecessors(self, s: MultiAgentState):
        ret = set()
        for action_tuple in itertools.product(SingleAgentAction, repeat=self.n_agents):
            action = MultiAgentAction({self.agents[i]: action_tuple[i]
                                       for i in range(len(action_tuple))})
            transitions = self._get_transitions(s, action, True)
            next_states = [t[1] for t in transitions]
            ret = ret.union(next_states)

        return ret

    @functools.lru_cache(maxsize=None)
    def is_collision_transition(self, prev_state: MultiAgentState, next_state: MultiAgentState):
        """
        NOTE: this means that trailing is allowed (where an agent is at a location where a different agent was before)
        """
        is_regular_collision = any([len(next_state.who_is_here(next_state[agent])) > 1 for agent in self.agents])
        is_switch_collision = any([all([agent1 != agent2,
                                        prev_state[agent1] == next_state[agent2],
                                        prev_state[agent2] == next_state[agent1]])
                                   for agent1, agent2 in itertools.combinations(self.agents, r=2)])

        return is_switch_collision or is_regular_collision

    # TODO: use this function for get_transitions
    @functools.lru_cache(maxsize=None)
    def single_agent_movements(self, local_state: SingleAgentState, local_action: SingleAgentAction):
        movements = []
        noises_prob = [self.fail_prob / 2, self.fail_prob / 2, 1 - self.fail_prob]
        for noise_idx in range(len(noises_prob)):
            prob = noises_prob[noise_idx]
            noised_action = NOISED_ACTIONS[local_action][noise_idx]
            next_state = self.grid.next_state(local_state, noised_action)
            movements.append((local_state, next_state, prob))

        return movements

    # Private methods
    def _living_reward(self, prev_state, next_state):
        if self.optimization_criteria == OptimizationCriteria.Makespan:
            return self.reward_of_living

        # SoC - an agent "pays" REWARD_OF_LIVING unless it reached its goal state and stayed there
        living_reward_per_agent = [
            0 if all([prev_state[agent] == self.goal_state[agent], next_state[agent] == self.goal_state[agent]])
            else self.reward_of_living
            for agent in self.agents
        ]
        return sum(living_reward_per_agent)

    @functools.lru_cache(maxsize=None)
    def _is_terminal_state(self, state: MultiAgentState):
        return any([
            all([state[agent] == self.goal_state[agent] for agent in self.agents]),  # goal
            any([len(state.who_is_here(state[agent])) > 1 for agent in self.agents])  # collision
        ])

        # "Private" methods

    def _reward_done_calculation(self, prev_state: MultiAgentState, next_state: MultiAgentState):
        # Calculate the reward of living in this step according to the optimization criteria
        living_reward = self._living_reward(prev_state, next_state)

        if self.is_collision_transition(prev_state, next_state):
            return living_reward + self.reward_of_collision, True

        # Check for a goal state
        if all([next_state[agent] == self.goal_state[agent] for agent in self.agents]):
            return living_reward + self.reward_of_goal, True

        return living_reward, False

    def _sample_noised_action(self, action: SingleAgentAction):
        noised_counter_clock_wise, noised_clock_wise, _ = NOISED_ACTIONS[action]
        half_fail_prob = self.fail_prob / 2
        noised_actions_and_probability = [
            (1 - self.fail_prob, action),
            (half_fail_prob, noised_clock_wise),
            (half_fail_prob, noised_counter_clock_wise)
        ]
        p = [x[0] for x in noised_actions_and_probability]

        noise_idx = categorical_sample(p, self.np_random)
        prob, new_action = noised_actions_and_probability[noise_idx]
        # prob, new_action = noised_actions_and_probability[
        #     np.random.choice(range(len(noised_actions_and_probability)), p=p)]

        return new_action, prob

    def _sample_next_state(self,
                           state: SingleAgentState,
                           action: SingleAgentAction):
        new_action, prob = self._sample_noised_action(action)
        return self.grid.next_state(state, new_action), prob

    @functools.lru_cache(maxsize=None)
    def _next_state(self, s: MultiAgentState, a: SingleAgentAction):
        return self.grid.next_state(s, a)

    @functools.lru_cache(maxsize=None)
    def _get_transitions(self, s: MultiAgentState, a: MultiAgentAction, for_predecessors=False):
        if self._is_terminal_state(s) and not for_predecessors:
            return [(1.0, s, 0, True)]

        # NOISED_ACTIONS last element is the same action (meaning, no slip happened)
        noises_prob = [self.fail_prob / 2, self.fail_prob / 2, 1 - self.fail_prob]
        next_state_to_transition = {}
        for noise_effect in itertools.product(filter(lambda noise_idx: noises_prob[noise_idx] > 0,
                                                     range(len(noises_prob))), repeat=self.n_agents):
            total_prob = functools.reduce(lambda x, y: x * noises_prob[y], noise_effect, 1)
            noised_action = MultiAgentAction({
                self.agents[i]: NOISED_ACTIONS[a.agent_to_action[self.agents[i]]][noise_effect[i]]
                for i in range(self.n_agents)
            })
            next_state = MultiAgentState({
                self.agents[i]: self._next_state(s.agent_to_state[self.agents[i]],
                                                 noised_action.agent_to_action[self.agents[i]])
                for i in range(self.n_agents)
            }, self.grid)
            reward, done = self._reward_done_calculation(s, next_state)
            if next_state in next_state_to_transition:
                curr_prob = next_state_to_transition[next_state][0]
                next_state_to_transition[next_state] = (total_prob + curr_prob, next_state, reward, done)
            else:
                next_state_to_transition[next_state] = (total_prob, next_state, reward, done)

        return next_state_to_transition.values()

    @functools.lru_cache(maxsize=None)
    def _get_transitions_curried(self, s: MultiAgentState):
        return function_to_get_item_of_object(functools.partial(self._get_transitions, s))


if __name__ == '__main__':
    pass
