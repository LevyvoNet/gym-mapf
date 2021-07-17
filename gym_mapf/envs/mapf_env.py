import random
import gym
import numpy as np
import enum
from colorama import Fore

from gym_mapf.envs.grid import (EmptyCell,
                                ObstacleCell,
                                MapfGrid,
                                MultiAgentState,
                                MultiAgentAction,
                                SingleAgentState,
                                SingleAgentAction,
                                MultiAgentStateSpace,
                                MultiAgentActionSpace,
                                ACTIONS,
                                CELL_TO_CHAR)

GYM_MAPF_SEED = 42

NOISED_ACTIONS = {
    SingleAgentAction.UP: (SingleAgentAction.RIGHT, SingleAgentAction.LEFT),
    SingleAgentAction.DOWN: (SingleAgentAction.LEFT, SingleAgentAction.RIGHT),
    SingleAgentAction.LEFT: (SingleAgentAction.UP, SingleAgentAction.DOWN),
    SingleAgentAction.RIGHT: (SingleAgentAction.DOWN, SingleAgentAction.UP),
    SingleAgentAction.STAY: (SingleAgentAction.STAY, SingleAgentAction.STAY)
}


def sample_noised_action(action: SingleAgentAction, fail_prob: float):
    noised_counter_clock_wise, noised_clock_wise = NOISED_ACTIONS[action]
    half_fail_prob = fail_prob / 2
    noised_actions_and_probability = [
        (1 - fail_prob, action),
        (half_fail_prob, noised_clock_wise),
        (half_fail_prob, noised_counter_clock_wise)
    ]
    p = [x[0] for x in noised_actions_and_probability]

    prob, new_action = np.random.choice(noised_actions_and_probability, p=p)

    return new_action, prob


def sample_next_state(grid: MapfGrid,
                      state: SingleAgentState,
                      action: SingleAgentAction,
                      fail_prob: int):
    new_action, prob = sample_noised_action(action, fail_prob)
    return grid.next_state(state, new_action), prob


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
                 fail_prob: int,
                 reward_of_clash: int,
                 reward_of_goal: int,
                 reward_of_living: int,
                 optimization_criteria: OptimizationCriteria):
        # Seed random generators
        self.seed = GYM_MAPF_SEED
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
        self.reward_of_collision = reward_of_clash
        self.reward_of_goal = reward_of_goal
        self.reward_of_living = reward_of_living

        # Initialize action and observation spaces
        self.action_space = MultiAgentActionSpace(n_agents)
        self.observation_space = MultiAgentStateSpace(n_agents, self.grid)

        # TODO: Is these fields really required?
        self.nS = len(self.grid.valid_locations) ** self.n_agents  # each agent may be in each of the cells.
        self.nA = len(ACTIONS) ** self.n_agents

        self.reset()

    def _reward_done_calculation(self, prev_state: MultiAgentState, next_state: MultiAgentState):
        # Check for a regular collision
        for _, state in next_state:
            if len(next_state.who_is_here(state) >= 2):
                return self.reward_of_collision, True

        # Check for a switch (also considered as a collision)
        for agent1 in prev_state.agents():
            for agent2 in prev_state.agents():
                if all([
                    agent1 != agent2,
                    prev_state[agent1] == next_state[agent2],
                    prev_state[agent2] == next_state[agent1]
                ]):
                    return self.reward_of_collision, True

        # Check for a goal state
        if all([
            next_state[agent] == self.goal_state[agent]
            for agent in prev_state.agents()
        ]):
            return self.reward_of_goal, True

        # Just a "regular" transition, return the cost of it regarding to the optimization criteria
        reward = {
            OptimizationCriteria.SoC: self.n_agents * self.reward_of_living,
            OptimizationCriteria.Makespan: self.reward_of_living
        }[self.optimization_criteria]

        return reward, False

    def step(self, action: MultiAgentAction):
        # Sample the next state and its probability
        agent_to_next_state = {}
        total_prob = 1
        for agent, state in self.s:
            next_state, prob = sample_next_state(self.grid, state, action[agent], self.fail_prob)
            total_prob *= prob
            agent_to_next_state[agent] = next_state
        next_state = MultiAgentState(agent_to_next_state)

        # Calculate the reward for the transition
        reward, done = self._reward_done_calculation(self.s, next_state)

        self.s = next_state
        return next_state, reward, done, {"prob": total_prob}

    def reset(self):
        self.s = self.start_state
        return self.s

    def render(self, mode='characters'):
        # v_state = self.state_to_locations(self.s)
        # v_agent_goals = self.agents_goals

        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                curr_state = SingleAgentState(i, j)
                if curr_state in self.s:
                    agents_in_state = self.s.who_is_here(curr_state)
                    assert len(agents_in_state) >= 1
                    if len(agents_in_state) > 1:
                        # There is another agent in this cell, a collision happened
                        print(Fore.RED + '*' + Fore.RESET, end=' ')
                    else:
                        # This is the only agent in this cell
                        if curr_state in self.goal_state and agents_in_state[0] == self.goal_state[agents_in_state[0]]:
                            # print an agent which reached it's goal
                            print(Fore.GREEN + str(agents_in_state[0]) + Fore.RESET, end=' ')
                            continue
                        print(Fore.YELLOW + str(agents_in_state[0]) + Fore.RESET, end=' ')
                    continue
                if curr_state in self.goal_state:
                    print(Fore.BLUE + str(self.goal_state.who_is_here(curr_state)) + Fore.RESET, end=' ')
                    continue

                print(CELL_TO_CHAR[self.grid[curr_state]], end=' ')

            print('')  # newline

    def close(self):
        super(self.__class__, self).close()

    def seed(self, seed=None):
        np.random.seed(self.seed)


if __name__ == '__main__':
    pass
