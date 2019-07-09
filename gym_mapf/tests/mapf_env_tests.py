import unittest
import os.path

from gym_mapf.envs.mapf_env import (MapfEnv,
                                    integer_state_to_vector,
                                    integer_action_to_vector,
                                    vector_state_to_integer,
                                    vector_action_to_integer)
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
        agent_starts = vector_state_to_integer(grid, ((0, 0), (7, 7)))
        agents_goals = vector_state_to_integer(grid, ((0, 2), (5, 7)))

        env = MapfEnv(grid, 2, agent_starts, agents_goals,
                      RIGHT_FAIL, LEFT_FAIL, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING)

        first_step_transitions = [(round(prob, 2), next_state, reward, done)
                                  for (prob, next_state, reward, done) in
                                  env.P[env.s][vector_action_to_integer((RIGHT, UP))]]

        self.assertEqual(set(first_step_transitions), {
            (0.64, vector_state_to_integer(grid, ((0, 1), (6, 7))), REWARD_OF_LIVING, False),  # (RIGHT, UP)
            (0.08, vector_state_to_integer(grid, ((1, 0), (6, 7))), REWARD_OF_LIVING, False),  # (DOWN, UP)
            (0.08, vector_state_to_integer(grid, ((0, 0), (6, 7))), REWARD_OF_LIVING, False),  # (UP, UP)
            (0.08, vector_state_to_integer(grid, ((0, 1), (7, 7))), REWARD_OF_LIVING, False),  # (RIGHT, RIGHT)
            (0.08, vector_state_to_integer(grid, ((0, 1), (7, 6))), REWARD_OF_LIVING, False),  # (RIGHT, LEFT)
            (0.01, vector_state_to_integer(grid, ((1, 0), (7, 7))), REWARD_OF_LIVING, False),  # (DOWN, RIGHT)
            (0.01, vector_state_to_integer(grid, ((1, 0), (7, 6))), REWARD_OF_LIVING, False),  # (DOWN, LEFT)
            (0.01, vector_state_to_integer(grid, ((0, 0), (7, 7))), REWARD_OF_LIVING, False),  # (UP, RIGHT)
            (0.01, vector_state_to_integer(grid, ((0, 0), (7, 6))), REWARD_OF_LIVING, False)  # (UP, LEFT)
        })

        wish_state = vector_state_to_integer(grid, ((0, 1), (6, 7)))
        second_step_transitions = [(round(prob, 2), next_state, reward, done)
                                   for (prob, next_state, reward, done) in
                                   env.P[wish_state][vector_action_to_integer((RIGHT, UP))]]

        # [(0,0), (7,7)]
        self.assertEqual(set(second_step_transitions), {
            (0.64, vector_state_to_integer(grid, ((0, 2), (5, 7))), REWARD_OF_GOAL, True),  # (RIGHT, UP)
            (0.08, vector_state_to_integer(grid, ((1, 1), (5, 7))), REWARD_OF_LIVING, False),  # (DOWN, UP)
            (0.08, vector_state_to_integer(grid, ((0, 1), (5, 7))), REWARD_OF_LIVING, False),  # (UP, UP)
            (0.08, vector_state_to_integer(grid, ((0, 2), (6, 7))), REWARD_OF_LIVING, False),  # (RIGHT, RIGHT)
            (0.08, vector_state_to_integer(grid, ((0, 2), (6, 6))), REWARD_OF_LIVING, False),  # (RIGHT, LEFT)
            (0.01, vector_state_to_integer(grid, ((1, 1), (6, 7))), REWARD_OF_LIVING, False),  # (DOWN, RIGHT)
            (0.01, vector_state_to_integer(grid, ((1, 1), (6, 6))), REWARD_OF_LIVING, False),  # (DOWN, LEFT)
            (0.01, vector_state_to_integer(grid, ((0, 1), (6, 7))), REWARD_OF_LIVING, False),  # (UP, RIGHT)
            (0.01, vector_state_to_integer(grid, ((0, 1), (6, 6))), REWARD_OF_LIVING, False)  # (UP, LEFT)
        })

    def test_colliding_agents_state_is_terminal_and_negative_reward(self):
        map_file_path = os.path.abspath(os.path.join(__file__, '../../maps/empty-8-8/empty-8-8.map'))

        grid = MapfGrid(parse_map_file(map_file_path))

        # agents are starting a
        agent_starts = vector_state_to_integer(grid, ((0, 0), (0, 2)))
        agents_goals = vector_state_to_integer(grid, ((7, 7), (5, 5)))

        env = MapfEnv(grid, 2, agent_starts, agents_goals,
                      RIGHT_FAIL, LEFT_FAIL, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING)
        transitions = [(round(prob, 2), next_state, reward, done)
                       for (prob, next_state, reward, done)
                       in env.P[env.s][vector_action_to_integer((RIGHT, LEFT))]]

        self.assertIn((0.64, vector_state_to_integer(grid, ((0, 1), (0, 1))), REWARD_OF_CLASH, True),
                      set(transitions))

    def test_agent_doesnt_move_if_reach_to_goal(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....'])

        # one agent is already at it's goal
        agent_starts = vector_state_to_integer(grid, ((0, 0), (3, 3)))
        agents_goals = vector_state_to_integer(grid, ((0, 0), (1, 3)))

        env = MapfEnv(grid, 2, agent_starts, agents_goals,
                      RIGHT_FAIL, LEFT_FAIL, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING)

        transitions = [(round(prob, 2), next_state, reward, done)
                       for (prob, next_state, reward, done)
                       in env.P[env.s][vector_action_to_integer((RIGHT, UP))]]

        self.assertEqual(set(transitions), {
            (0.8, vector_state_to_integer(grid, ((0, 0), (2, 3))), REWARD_OF_LIVING, False),  # (STAY, UP)
            (0.1, vector_state_to_integer(grid, ((0, 0), (3, 3))), REWARD_OF_LIVING, False),  # (STAY, RIGHT)
            (0.1, vector_state_to_integer(grid, ((0, 0), (3, 2))), REWARD_OF_LIVING, False),  # (STAY, LEFT)
        })

    def test_soc_makespan(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....'])

        agent_starts = vector_state_to_integer(grid, ((0, 0), (3, 3), (1, 1)))
        agents_goals = vector_state_to_integer(grid, ((0, 1), (1, 3), (1, 2)))

        determinstic_env = MapfEnv(grid, 3, agent_starts, agents_goals,
                                   0.0, 0.0, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING)

        determinstic_env.step(vector_action_to_integer((RIGHT, UP, RIGHT)))
        s, r, done, _ = determinstic_env.step(vector_action_to_integer((RIGHT, UP, RIGHT)))

        self.assertEqual(s, vector_state_to_integer(grid, ((0, 1), (1, 3), (1, 2))))
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
        determinstic_env = MapfEnv(grid, 1, vector_state_to_integer(grid, ((0, 0),)),
                                   vector_state_to_integer(grid, ((4, 0),)),
                                   0.0, 0.0, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING)

        determinstic_env.step(vector_action_to_integer((DOWN,)))
        determinstic_env.step(vector_action_to_integer((DOWN,)))
        determinstic_env.step(vector_action_to_integer((DOWN,)))
        s, r, done, _ = determinstic_env.step(vector_action_to_integer((DOWN,)))

        self.assertEqual(s, vector_state_to_integer(grid, ((4, 0),)))
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
        env = MapfEnv(grid, 1, vector_state_to_integer(grid, ((0, 0),)),
                      vector_state_to_integer(grid, ((4, 0),)),
                      0.0, 0.0, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING)

        self.assertEqual(env.soc, 0)
        self.assertEqual(env.makespan, 0)

        env.step(vector_action_to_integer((RIGHT,)))

        self.assertEqual(env.soc, 1)
        self.assertEqual(env.makespan, 1)

        env_copy = copy(env)
        env_copy.step(vector_action_to_integer((RIGHT,)))

        self.assertEqual(env.soc, 1)
        self.assertEqual(env.makespan, 1)
        self.assertEqual(env_copy.soc, 2)
        self.assertEqual(env_copy.makespan, 2)

    def test_action_from_terminal_state_has_no_effect(self):
        grid = MapfGrid(['..',
                         '..'])
        env = MapfEnv(grid, 1, vector_state_to_integer(grid, ((0, 0),)),
                      vector_state_to_integer(grid, ((1, 1),)),
                      0.0, 0.0, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING)

        state, reward, done, _ = env.step(vector_action_to_integer((RIGHT,)))
        self.assertEqual(reward, 0)
        self.assertEqual(done, False)
        state, reward, done, _ = env.step(vector_action_to_integer((DOWN,)))
        self.assertEqual(reward, REWARD_OF_GOAL)
        self.assertEqual(done, True)
        # now, after the game is finished - do another step and make sure it has not effect.
        state_after_done, reward_after_done, done_after_done, _ = env.step(vector_action_to_integer((UP,)))
        self.assertEqual(state_after_done, state)
        self.assertEqual(done_after_done, True)
        self.assertEqual(reward_after_done, 0)
        # another time like I'm trying to reach the goal
        state_after_done, reward_after_done, done_after_done, _ = env.step(vector_action_to_integer((DOWN,)))
        self.assertEqual(state_after_done, state)
        self.assertEqual(done_after_done, True)
        self.assertEqual(reward_after_done, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
