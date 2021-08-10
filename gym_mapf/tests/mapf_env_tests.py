import unittest
import os.path

from gym_mapf.envs.mapf_env import MapfEnv, OptimizationCriteria, MultiAgentAction, MultiAgentState
from gym_mapf.envs.grid import (MapfGrid,
                                SingleAgentState,
                                SingleAgentAction)
from gym_mapf.envs.utils import parse_map_file, create_mapf_env
from gym_mapf.tests import MAPS_DIR
from copy import copy

FAIL_PROB = 0.2
REWARD_OF_CLASH = -1000
REWARD_OF_LIVING = -1
REWARD_OF_GOAL = 100


class MapfEnvTest(unittest.TestCase):
    def test_transition_function_empty_grid(self):
        """Assert the basic steps are done right.

        * Define an empty 8x8 environment with two agents starting at (0,0),(7,7) and desire to reach (0,2),(5,7).
        * Perform one (RIGHT, UP) step and assert that the transitions are correct.
        * Perform another (RIGHT, UP) step from the most probable next state from before ((0,1), (6,7)) and assert
            that the transitions are correct again, including the terminal one.
        """
        map_file_path = os.path.abspath(os.path.join(__file__, MAPS_DIR, 'empty-8-8/empty-8-8.map'))
        grid = MapfGrid(parse_map_file(map_file_path))

        # agents are starting a
        start_state = MultiAgentState({
            0: SingleAgentState(0, 0),
            1: SingleAgentState(7, 7)
        }, grid)

        goal_state = MultiAgentState({
            0: SingleAgentState(0, 2),
            1: SingleAgentState(5, 7)
        }, grid)

        env = MapfEnv(grid, 2, start_state, goal_state, 0.2, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING,
                      OptimizationCriteria.Makespan)
        right_up_action = MultiAgentAction({0: SingleAgentAction.RIGHT, 1: SingleAgentAction.UP})
        first_step_transitions = [(round(prob, 2), next_state, reward, done)
                                  for (prob, next_state, reward, done) in
                                  env.P[env.s][right_up_action]]

        self.assertEqual(set(first_step_transitions), {
            (0.64,
             MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(6, 7)}, grid),
             REWARD_OF_LIVING,
             False),  # (RIGHT, UP)
            (0.08,
             MultiAgentState({0: SingleAgentState(1, 0), 1: SingleAgentState(6, 7)}, grid),
             REWARD_OF_LIVING,
             False),  # (DOWN, UP)
            (0.08,
             MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(6, 7)}, grid),
             REWARD_OF_LIVING,
             False),  # (UP, UP)
            (0.08,
             MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(7, 7)}, grid),
             REWARD_OF_LIVING,
             False),  # (RIGHT, RIGHT)
            (0.08,
             MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(7, 6)}, grid),
             REWARD_OF_LIVING,
             False),  # (RIGHT, LEFT)
            (0.01,
             MultiAgentState({0: SingleAgentState(1, 0), 1: SingleAgentState(7, 7)}, grid),
             REWARD_OF_LIVING,
             False),  # (DOWN, RIGHT)
            (0.01,
             MultiAgentState({0: SingleAgentState(1, 0), 1: SingleAgentState(7, 6)}, grid),
             REWARD_OF_LIVING,
             False),  # (DOWN, LEFT)
            (0.01,
             MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(7, 7)}, grid),
             REWARD_OF_LIVING,
             False),  # (UP, RIGHT)
            (0.01,
             MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(7, 6)}, grid),
             REWARD_OF_LIVING,
             False)  # (UP, LEFT)
        })

        wish_state = MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(6, 7)}, grid)
        second_step_transitions = [(round(prob, 2), next_state, reward, done)
                                   for (prob, next_state, reward, done) in
                                   env.P[wish_state][right_up_action]]

        # [(0,0), (7,7)]
        self.assertEqual(set(second_step_transitions), {
            (0.64,
             MultiAgentState({0: SingleAgentState(0, 2), 1: SingleAgentState(5, 7)}, grid),
             REWARD_OF_LIVING + REWARD_OF_GOAL,
             True),  # (RIGHT, UP)
            (0.08,
             MultiAgentState({0: SingleAgentState(1, 1), 1: SingleAgentState(5, 7)}, grid),
             REWARD_OF_LIVING,
             False),  # (DOWN, UP)
            (0.08,
             MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(5, 7)}, grid),
             REWARD_OF_LIVING,
             False),  # (UP, UP)
            (0.08,
             MultiAgentState({0: SingleAgentState(0, 2), 1: SingleAgentState(6, 7)}, grid),
             REWARD_OF_LIVING,
             False),  # (RIGHT, RIGHT)
            (0.08,
             MultiAgentState({0: SingleAgentState(0, 2), 1: SingleAgentState(6, 6)}, grid),
             REWARD_OF_LIVING,
             False),  # (RIGHT, LEFT)
            (0.01,
             MultiAgentState({0: SingleAgentState(1, 1), 1: SingleAgentState(6, 7)}, grid),
             REWARD_OF_LIVING,
             False),  # (DOWN, RIGHT)
            (0.01,
             MultiAgentState({0: SingleAgentState(1, 1), 1: SingleAgentState(6, 6)}, grid),
             REWARD_OF_LIVING,
             False),  # (DOWN, LEFT)
            (0.01,
             MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(6, 7)}, grid),
             REWARD_OF_LIVING,
             False),  # (UP, RIGHT)
            (0.01,
             MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(6, 6)}, grid),
             REWARD_OF_LIVING,
             False)  # (UP, LEFT)
        })

    def test_colliding_agents_state_is_terminal_and_negative_reward(self):
        map_file_path = os.path.abspath(os.path.join(__file__, MAPS_DIR, 'empty-8-8/empty-8-8.map'))

        grid = MapfGrid(parse_map_file(map_file_path))

        # agents are starting a
        start_state = MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 2)}, grid)
        goal_state = MultiAgentState({0: SingleAgentState(7, 7), 1: SingleAgentState(5, 5)}, grid)

        env = MapfEnv(grid, 2, start_state, goal_state, FAIL_PROB, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING,
                      OptimizationCriteria.Makespan)
        right_left_action = MultiAgentAction({0: SingleAgentAction.RIGHT, 1: SingleAgentAction.LEFT})
        transitions = [(round(prob, 2), next_state, reward, done)
                       for (prob, next_state, reward, done)
                       in env.P[env.s][right_left_action]]

        self.assertIn((0.64,
                       MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(0, 1)}, grid),
                       REWARD_OF_LIVING + REWARD_OF_CLASH,
                       True),
                      transitions)

    def test_reward_multiagent_soc(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....'])

        start_state = MultiAgentState({0: SingleAgentState(0, 0),
                                       1: SingleAgentState(3, 3),
                                       2: SingleAgentState(1, 1)}, grid)
        goal_state = MultiAgentState({0: SingleAgentState(0, 1),
                                      1: SingleAgentState(1, 3),
                                      2: SingleAgentState(1, 2)}, grid)

        determinstic_env = MapfEnv(grid, 3, start_state, goal_state, 0, REWARD_OF_CLASH, REWARD_OF_GOAL,
                                   REWARD_OF_LIVING,
                                   OptimizationCriteria.SoC)
        total_reward = 0
        s, r, done, _ = determinstic_env.step(MultiAgentAction({
            0: SingleAgentAction.RIGHT,
            1: SingleAgentAction.UP,
            2: SingleAgentAction.RIGHT,
        }))
        total_reward += r
        self.assertFalse(done)

        s, r, done, _ = determinstic_env.step(MultiAgentAction({
            0: SingleAgentAction.STAY,
            1: SingleAgentAction.UP,
            2: SingleAgentAction.STAY,
        }))
        total_reward += r
        self.assertEqual(s, goal_state)
        self.assertTrue(done)
        self.assertEqual(total_reward, 4 * REWARD_OF_LIVING + REWARD_OF_GOAL)

    def test_reawrd_multiagent_makespan(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....'])

        start_state = MultiAgentState({0: SingleAgentState(0, 0),
                                       1: SingleAgentState(3, 3),
                                       2: SingleAgentState(1, 1)}, grid)
        goal_state = MultiAgentState({0: SingleAgentState(0, 1),
                                      1: SingleAgentState(1, 3),
                                      2: SingleAgentState(1, 2)}, grid)

        determinstic_env = MapfEnv(grid, 3, start_state, goal_state, 0, REWARD_OF_CLASH, REWARD_OF_GOAL,
                                   REWARD_OF_LIVING,
                                   OptimizationCriteria.Makespan)

        total_reward = 0
        s, r, done, _ = determinstic_env.step(MultiAgentAction({
            0: SingleAgentAction.RIGHT,
            1: SingleAgentAction.UP,
            2: SingleAgentAction.RIGHT,
        }))
        total_reward += r
        self.assertFalse(done)

        s, r, done, _ = determinstic_env.step(MultiAgentAction({
            0: SingleAgentAction.STAY,
            1: SingleAgentAction.UP,
            2: SingleAgentAction.STAY,
        }))
        total_reward += r
        self.assertEqual(s, goal_state)
        self.assertTrue(done)
        self.assertEqual(total_reward, 2 * REWARD_OF_LIVING + REWARD_OF_GOAL)

    def test_reward_single_agent_soc(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....',
            '....'])

        start_state = MultiAgentState({
            0: SingleAgentState(0, 0),
        }, grid)
        goal_state = MultiAgentState({
            0: SingleAgentState(4, 0),
        }, grid)

        determinstic_env = MapfEnv(grid, 1, start_state, goal_state, 0, REWARD_OF_CLASH, REWARD_OF_GOAL,
                                   REWARD_OF_LIVING, OptimizationCriteria.SoC)
        total_reward = 0
        down_action = MultiAgentAction({0: SingleAgentAction.DOWN})
        _, r, _, _ = determinstic_env.step(down_action)
        total_reward += r
        _, r, _, _ = determinstic_env.step(down_action)
        total_reward += r
        _, r, _, _ = determinstic_env.step(down_action)
        total_reward += r
        s, r, done, _ = determinstic_env.step(down_action)
        total_reward += r

        self.assertEqual(s, goal_state)
        self.assertEqual(r, REWARD_OF_LIVING + REWARD_OF_GOAL)
        self.assertEqual(total_reward, REWARD_OF_GOAL + 4 * REWARD_OF_LIVING)

    def test_reward_single_agent_makespan(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....',
            '....'])

        start_state = MultiAgentState({
            0: SingleAgentState(0, 0),
        }, grid)
        goal_state = MultiAgentState({
            0: SingleAgentState(4, 0),
        }, grid)

        determinstic_env = MapfEnv(grid, 1, start_state, goal_state, 0, REWARD_OF_CLASH, REWARD_OF_GOAL,
                                   REWARD_OF_LIVING, OptimizationCriteria.Makespan)
        total_reward = 0
        down_action = MultiAgentAction({0: SingleAgentAction.DOWN})
        _, r, _, _ = determinstic_env.step(down_action)
        total_reward += r
        _, r, _, _ = determinstic_env.step(down_action)
        total_reward += r
        _, r, _, _ = determinstic_env.step(down_action)
        total_reward += r
        s, r, done, _ = determinstic_env.step(down_action)
        total_reward += r

        self.assertEqual(s, goal_state)
        self.assertEqual(r, REWARD_OF_LIVING + REWARD_OF_GOAL)

        self.assertEqual(total_reward, REWARD_OF_GOAL + 4 * REWARD_OF_LIVING)

    def test_copy_mapf_env(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....',
            '....'])

        start_state = MultiAgentState({0: SingleAgentState(0, 0)}, grid)
        goal_state = MultiAgentState({0: SingleAgentState(4, 0)}, grid)

        env = MapfEnv(grid, 1, start_state, goal_state, 0, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING,
                      OptimizationCriteria.Makespan)

        right_action = MultiAgentAction({0: SingleAgentAction.RIGHT})
        env.step(right_action)
        state_after_single_right = MultiAgentState({0: SingleAgentState(0, 1)}, grid)
        self.assertEqual(env.s, state_after_single_right)

        env_copy = copy(env)
        self.assertEqual(env_copy.s, state_after_single_right)
        env_copy.step(right_action)
        state_after_two_rights = MultiAgentState({0: SingleAgentState(0, 2)}, grid)
        self.assertEqual(env_copy.s, state_after_two_rights)

        # Make sure the original env did not change
        self.assertEqual(env.s, state_after_single_right)

    def test_action_from_terminal_state_has_no_effect(self):
        grid = MapfGrid(['..',
                         '..'])
        start_state = MultiAgentState({0: SingleAgentState(0, 0)}, grid)
        goal_state = MultiAgentState({0: SingleAgentState(1, 1)}, grid)

        env = MapfEnv(grid, 1, start_state, goal_state, 0, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING,
                      OptimizationCriteria.Makespan)

        state, reward, done, _ = env.step(MultiAgentAction({0: SingleAgentAction.RIGHT}))
        self.assertEqual(reward, REWARD_OF_LIVING)
        self.assertEqual(done, False)
        state, reward, done, _ = env.step(MultiAgentAction({0: SingleAgentAction.DOWN}))
        self.assertEqual(reward, REWARD_OF_LIVING + REWARD_OF_GOAL)
        self.assertEqual(done, True)

        # now, after the game is finished - do another step and make sure it has not effect.
        state_after_done, reward_after_done, done_after_done, _ = env.step(MultiAgentAction({0: SingleAgentAction.UP}))
        self.assertEqual(state_after_done, state)
        self.assertEqual(done_after_done, True)
        self.assertEqual(reward_after_done, 0)

        # another time like I'm trying to reach the goal
        state_after_done, reward_after_done, done_after_done, _ = env.step(
            MultiAgentAction({0: SingleAgentAction.DOWN}))
        self.assertEqual(state_after_done, state)
        self.assertEqual(done_after_done, True)
        self.assertEqual(reward_after_done, 0)

    def test_switch_spots_is_a_collision(self):
        grid = MapfGrid(['..'])

        start_state = MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 1)}, grid)
        goal_state = MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(0, 0)}, grid)

        deterministic_env = MapfEnv(grid, 2, start_state, goal_state, 0, REWARD_OF_CLASH, REWARD_OF_GOAL,
                                    REWARD_OF_LIVING,
                                    OptimizationCriteria.Makespan)

        s, r, done, _ = deterministic_env.step(
            MultiAgentAction({0: SingleAgentAction.RIGHT, 1: SingleAgentAction.LEFT}))

        # Assert the game terminated in a collision
        self.assertEqual(done, True)
        self.assertEqual(r, REWARD_OF_LIVING + REWARD_OF_CLASH)

    def test_predecessors(self):
        """Assert the predecessors function works correctly.

        Create an environment which looks like that:
        ....
        ..0.
        .1..

        3X4 grid.
        agent 0 is at (1,2)
        agent 1 is at (2,1)

        The predecessors for agent 0 are:
        1. (0,2)
        2. (1,1)
        3. (1,3)
        4. (2,2)

        The predecessors for agent 1 are:
        1. (2,2)
        2. (2,0)
        3. (1,1)

        Therefore, the predecessors states of the initial state corresponds to these locations:
        1.  ((0,2), (2,2))
        2.  ((0,2), (2,0))
        3.  ((0,2), (1,1))
        4.  ((0,2), (2,1))
        5.  ((1,1), (2,2))
        6.  ((1,1), (2,0))
        7.  ((1,1), (1,1))
        8.  ((1,1), (2,1))
        9.  ((1,3), (2,2))
        10. ((1,3), (2,0))
        11. ((1,3), (1,1))
        12. ((1,3), (2,1))
        13. ((2,2), (2,2))
        14. ((2,2), (2,0))
        15. ((2,2), (1,1))
        16. ((2,2), (2,1))
        17. ((1,2), (2,2))
        18. ((1,2), (2,0))
        19. ((1,2), (1,1))
        20. ((1,2), (2,1))
        """
        grid = MapfGrid(['....',
                         '....',
                         '....'])

        start_state = MultiAgentState({0: SingleAgentState(1, 2), 1: SingleAgentState(2, 1)}, grid)
        # don't care
        goal_state = MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(2, 3)}, grid)

        env = MapfEnv(grid, 2, start_state, goal_state, 0, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING,
                      OptimizationCriteria.Makespan)

        expected_locations = [
            ((0, 2), (2, 2)),
            ((0, 2), (2, 0)),
            ((0, 2), (1, 1)),  # bug
            ((0, 2), (2, 1)),

            ((1, 1), (2, 2)),
            ((1, 1), (2, 0)),  # bug
            ((1, 1), (1, 1)),
            ((1, 1), (2, 1)),

            ((1, 3), (2, 2)),  # bug
            ((1, 3), (2, 0)),
            ((1, 3), (1, 1)),
            ((1, 3), (2, 1)),

            ((2, 2), (2, 2)),
            ((2, 2), (2, 0)),
            ((2, 2), (1, 1)),
            ((2, 2), (2, 1)),

            ((1, 2), (2, 2)),
            ((1, 2), (2, 0)),
            ((1, 2), (1, 1)),
            ((1, 2), (2, 1))
        ]

        expected_states = [MultiAgentState({0: SingleAgentState(*loc[0]), 1: SingleAgentState(*loc[1])}, grid)
                           for loc in expected_locations]

        self.assertEqual(set(expected_states),
                         set(env.predecessors(env.s)))

    def test_similar_transitions_probability_summed(self):
        grid = MapfGrid(['..',
                         '..'])
        start_state = MultiAgentState({0: SingleAgentState(0, 0)}, grid)
        goal_state = MultiAgentState({0: SingleAgentState(1, 1)}, grid)
        env = MapfEnv(grid, 1, start_state, goal_state, 0.1, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING,
                      OptimizationCriteria.Makespan)

        self.assertEqual([*env.P[env.s][MultiAgentAction({0: SingleAgentAction.STAY})]],
                         [(1.0, env.s, REWARD_OF_LIVING, False)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
