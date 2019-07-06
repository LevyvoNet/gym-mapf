import unittest
import os.path

from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.mapf.grid import MapfGrid
from gym_mapf.envs.utils import parse_map_file
from gym_mapf.envs import *
from copy import copy

RIGHT_FAIL = 0.1
LEFT_FAIL = 0.1
REWARD_OF_CLASH = -1000.0
REWARD_OF_LIVING = 0.0
REWARD_OF_GOAL = 100.0


class MapfEnvTest(unittest.TestCase):
    def test_transition_function_empty_grid(self):
        """Assert the basic steps are done right.

        * Define an empty 8x8 environment with two agents starting at (0,0),(7,7) and desire to reach (0,2),(5,7).
        * Perform one (RIGHT, UP) step and assert that the transitions are correct.
        * Perform another (RIGHT, UP) step from the most probable next state from before ((0,1), (6,7)) and assert
            that the transitions are correct again, including the terminal one.
        """
        map_file_path = os.path.abspath(os.path.join(__file__, '../../maps/empty-8-8/empty-8-8.map'))
        grid = MapfGrid(parse_map_file(map_file_path))

        # agents are starting a
        agent_starts, agents_goals = ((0, 0), (7, 7)), ((0, 2), (5, 7))
        env = MapfEnv(grid, agent_starts, agents_goals,
                      RIGHT_FAIL, LEFT_FAIL, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING)

        first_step_transitions = [(round(prob, 2), next_state, reward, done)
                                  for (prob, next_state, reward, done) in env.P[env.s][(RIGHT, UP)]]

        self.assertEqual(set(first_step_transitions), {
            (0.64, ((0, 1), (6, 7)), REWARD_OF_LIVING, False),  # (RIGHT, UP)
            (0.08, ((1, 0), (6, 7)), REWARD_OF_LIVING, False),  # (DOWN, UP)
            (0.08, ((0, 0), (6, 7)), REWARD_OF_LIVING, False),  # (UP, UP)
            (0.08, ((0, 1), (7, 7)), REWARD_OF_LIVING, False),  # (RIGHT, RIGHT)
            (0.08, ((0, 1), (7, 6)), REWARD_OF_LIVING, False),  # (RIGHT, LEFT)
            (0.01, ((1, 0), (7, 7)), REWARD_OF_LIVING, False),  # (DOWN, RIGHT)
            (0.01, ((1, 0), (7, 6)), REWARD_OF_LIVING, False),  # (DOWN, LEFT)
            (0.01, ((0, 0), (7, 7)), REWARD_OF_LIVING, False),  # (UP, RIGHT)
            (0.01, ((0, 0), (7, 6)), REWARD_OF_LIVING, False)  # (UP, LEFT)
        })

        wish_state = ((0, 1), (6, 7))
        second_step_transitions = [(round(prob, 2), next_state, reward, done)
                                   for (prob, next_state, reward, done) in env.P[wish_state][(RIGHT, UP)]]

        # [(0,0), (7,7)]
        self.assertEqual(set(second_step_transitions), {
            (0.64, ((0, 2), (5, 7)), REWARD_OF_GOAL, True),  # (RIGHT, UP)
            (0.08, ((1, 1), (5, 7)), REWARD_OF_LIVING, False),  # (DOWN, UP)
            (0.08, ((0, 1), (5, 7)), REWARD_OF_LIVING, False),  # (UP, UP)
            (0.08, ((0, 2), (6, 7)), REWARD_OF_LIVING, False),  # (RIGHT, RIGHT)
            (0.08, ((0, 2), (6, 6)), REWARD_OF_LIVING, False),  # (RIGHT, LEFT)
            (0.01, ((1, 1), (6, 7)), REWARD_OF_LIVING, False),  # (DOWN, RIGHT)
            (0.01, ((1, 1), (6, 6)), REWARD_OF_LIVING, False),  # (DOWN, LEFT)
            (0.01, ((0, 1), (6, 7)), REWARD_OF_LIVING, False),  # (UP, RIGHT)
            (0.01, ((0, 1), (6, 6)), REWARD_OF_LIVING, False)  # (UP, LEFT)
        })

    def test_colliding_agents_state_is_terminal_and_negative_reward(self):
        map_file_path = os.path.abspath(os.path.join(__file__, '../../maps/empty-8-8/empty-8-8.map'))

        grid = MapfGrid(parse_map_file(map_file_path))

        # agents are starting a
        agent_starts, agents_goals = ((0, 0), (0, 2)), ((7, 7), (5, 5))
        env = MapfEnv(grid, agent_starts, agents_goals,
                      RIGHT_FAIL, LEFT_FAIL, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING)
        transitions = [(round(prob, 2), next_state, reward, done)
                       for (prob, next_state, reward, done) in env.P[env.s][(RIGHT, LEFT)]]

        self.assertIn((0.64, ((0, 1), (0, 1)), REWARD_OF_CLASH, True), set(transitions))

    def test_agent_doesnt_move_if_reach_to_goal(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....'])

        agent_starts, agents_goals = ((0, 0), (3, 3)), ((0, 0), (1, 3))  # one agent is already at it's goal
        env = MapfEnv(grid, agent_starts, agents_goals,
                      RIGHT_FAIL, LEFT_FAIL, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING)

        transitions = [(round(prob, 2), next_state, reward, done)
                       for (prob, next_state, reward, done) in env.P[env.s][(RIGHT, UP)]]

        self.assertEqual(set(transitions), {
            (0.8, ((0, 0), (2, 3)), REWARD_OF_LIVING, False),  # (STAY, UP)
            (0.1, ((0, 0), (3, 3)), REWARD_OF_LIVING, False),  # (STAY, RIGHT)
            (0.1, ((0, 0), (3, 2)), REWARD_OF_LIVING, False),  # (STAY, LEFT)
        })

    def test_soc_makespan(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....'])

        agent_starts, agents_goals = ((0, 0), (3, 3), (1, 1)), ((0, 1), (1, 3), (1, 2))
        determinstic_env = MapfEnv(grid, agent_starts, agents_goals,
                                   0.0, 0.0, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING)

        determinstic_env.step((RIGHT, UP, RIGHT))
        s, r, done, _ = determinstic_env.step((RIGHT, UP, RIGHT))

        self.assertEqual(s, ((0, 1), (1, 3), (1, 2)))
        self.assertEqual(r, REWARD_OF_GOAL)
        self.assertEqual(determinstic_env.soc, 4)
        self.assertEqual(determinstic_env.makespan, 2)

    def test_soc_makespan_single_agent(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....',
            '....'])
        determinstic_env = MapfEnv(grid, ((0, 0),), ((4, 0),),
                                   0.0, 0.0, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING)

        determinstic_env.step((DOWN,))
        determinstic_env.step((DOWN,))
        determinstic_env.step((DOWN,))
        s, r, done, _ = determinstic_env.step((DOWN,))

        self.assertEqual(s, ((4, 0),))
        self.assertEqual(r, REWARD_OF_GOAL)
        self.assertEqual(determinstic_env.soc, 4)
        self.assertEqual(determinstic_env.makespan, 4)

    def test_copy_mapf_env(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....',
            '....'])
        env = MapfEnv(grid, ((0, 0),), ((4, 0),),
                      0.0, 0.0, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING)

        self.assertEqual(env.soc, 0)
        self.assertEqual(env.makespan, 0)

        env.step((RIGHT,))

        self.assertEqual(env.soc, 1)
        self.assertEqual(env.makespan, 1)

        env_copy = copy(env)
        env_copy.step((RIGHT,))

        self.assertEqual(env.soc, 1)
        self.assertEqual(env.makespan, 1)
        self.assertEqual(env_copy.soc, 2)
        self.assertEqual(env_copy.makespan, 2)

    def test_integer_states_and_actions(self):
        grid = MapfGrid([
            '...',
            '...',
            '...',
            '...'])

        agent_starts, agents_goals = ((0, 0), (3, 3)), ((0, 0), (1, 3))  # don't care
        env = MapfEnv(grid, agent_starts, agents_goals,
                      RIGHT_FAIL, LEFT_FAIL, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING)

        self.assertEqual(env.P[((3, 2), (0, 0))][(UP, UP)], env.P[10][0])
        self.assertEqual(env.P[((0, 1), (0, 1))][(UP, RIGHT)], env.P[13][5])


if __name__ == '__main__':
    unittest.main(verbosity=2)
