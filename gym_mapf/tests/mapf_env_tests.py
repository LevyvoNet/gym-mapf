import unittest
import os.path

from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    OptimizationCriteria)
from gym_mapf.envs.grid import MapfGrid
from gym_mapf.envs.utils import parse_map_file, create_mapf_env
from gym_mapf.envs import *
from gym_mapf.tests import MAPS_DIR
from copy import copy

FAIL_PROB = 0.2
REWARD_OF_CLASH = -1000.0
REWARD_OF_LIVING = -1.0
REWARD_OF_GOAL = 100.0


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
        agent_starts = ((0, 0), (7, 7))
        agents_goals = ((0, 2), (5, 7))

        env = MapfEnv(grid, 2, agent_starts, agents_goals,
                      FAIL_PROB, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING, OptimizationCriteria.Makespan)

        first_step_transitions = [((round(prob, 2), collision), next_state, reward, done)
                                  for ((prob, collision), next_state, reward, done) in
                                  env.P[env.s][vector_action_to_integer((RIGHT, UP))]]

        self.assertEqual(set(first_step_transitions), {
            ((0.64, False), env.locations_to_state(((0, 1), (6, 7))), REWARD_OF_LIVING, False),  # (RIGHT, UP)
            ((0.08, False), env.locations_to_state(((1, 0), (6, 7))), REWARD_OF_LIVING, False),  # (DOWN, UP)
            ((0.08, False), env.locations_to_state(((0, 0), (6, 7))), REWARD_OF_LIVING, False),  # (UP, UP)
            ((0.08, False), env.locations_to_state(((0, 1), (7, 7))), REWARD_OF_LIVING, False),  # (RIGHT, RIGHT)
            ((0.08, False), env.locations_to_state(((0, 1), (7, 6))), REWARD_OF_LIVING, False),  # (RIGHT, LEFT)
            ((0.01, False), env.locations_to_state(((1, 0), (7, 7))), REWARD_OF_LIVING, False),  # (DOWN, RIGHT)
            ((0.01, False), env.locations_to_state(((1, 0), (7, 6))), REWARD_OF_LIVING, False),  # (DOWN, LEFT)
            ((0.01, False), env.locations_to_state(((0, 0), (7, 7))), REWARD_OF_LIVING, False),  # (UP, RIGHT)
            ((0.01, False), env.locations_to_state(((0, 0), (7, 6))), REWARD_OF_LIVING, False)  # (UP, LEFT)
        })

        wish_state = env.locations_to_state(((0, 1), (6, 7)))
        second_step_transitions = [((round(prob, 2), collision), next_state, reward, done)
                                   for ((prob, collision), next_state, reward, done) in
                                   env.P[wish_state][vector_action_to_integer((RIGHT, UP))]]

        # [(0,0), (7,7)]
        self.assertEqual(set(second_step_transitions), {
            ((0.64, False), env.locations_to_state(((0, 2), (5, 7))), REWARD_OF_LIVING + REWARD_OF_GOAL, True),
            # (RIGHT, UP)
            ((0.08, False), env.locations_to_state(((1, 1), (5, 7))), REWARD_OF_LIVING, False),  # (DOWN, UP)
            ((0.08, False), env.locations_to_state(((0, 1), (5, 7))), REWARD_OF_LIVING, False),  # (UP, UP)
            ((0.08, False), env.locations_to_state(((0, 2), (6, 7))), REWARD_OF_LIVING, False),  # (RIGHT, RIGHT)
            ((0.08, False), env.locations_to_state(((0, 2), (6, 6))), REWARD_OF_LIVING, False),  # (RIGHT, LEFT)
            ((0.01, False), env.locations_to_state(((1, 1), (6, 7))), REWARD_OF_LIVING, False),  # (DOWN, RIGHT)
            ((0.01, False), env.locations_to_state(((1, 1), (6, 6))), REWARD_OF_LIVING, False),  # (DOWN, LEFT)
            ((0.01, False), env.locations_to_state(((0, 1), (6, 7))), REWARD_OF_LIVING, False),  # (UP, RIGHT)
            ((0.01, False), env.locations_to_state(((0, 1), (6, 6))), REWARD_OF_LIVING, False)  # (UP, LEFT)
        })

    def test_colliding_agents_state_is_terminal_and_negative_reward(self):
        map_file_path = os.path.abspath(os.path.join(__file__, MAPS_DIR, 'empty-8-8/empty-8-8.map'))

        grid = MapfGrid(parse_map_file(map_file_path))

        # agents are starting a
        agent_starts = ((0, 0), (0, 2))
        agents_goals = ((7, 7), (5, 5))

        env = MapfEnv(grid, 2, agent_starts, agents_goals,
                      FAIL_PROB, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING, OptimizationCriteria.Makespan)
        transitions = [((round(prob, 2), collision), next_state, reward, done)
                       for ((prob, collision), next_state, reward, done)
                       in env.P[env.s][vector_action_to_integer((RIGHT, LEFT))]]

        self.assertIn(
            ((0.64, True), env.locations_to_state(((0, 1), (0, 1))), REWARD_OF_LIVING + REWARD_OF_CLASH, True),
            set(transitions))

    def test_copy_mapf_env(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....',
            '....'])
        env = MapfEnv(grid, 1, ((0, 0),), ((4, 0),),
                      0, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING, OptimizationCriteria.Makespan)

        env.step(vector_action_to_integer((RIGHT,)))

        env_copy = copy(env)
        env_copy.step(vector_action_to_integer((RIGHT,)))

    def test_action_from_terminal_state_has_no_effect(self):
        grid = MapfGrid(['..',
                         '..'])
        env = MapfEnv(grid, 1, ((0, 0),), ((1, 1),),
                      0, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING, OptimizationCriteria.Makespan)

        state, reward, done, _ = env.step(vector_action_to_integer((RIGHT,)))
        self.assertEqual(reward, REWARD_OF_LIVING)
        self.assertEqual(done, False)
        state, reward, done, _ = env.step(vector_action_to_integer((DOWN,)))
        self.assertEqual(reward, REWARD_OF_LIVING + REWARD_OF_GOAL)
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

    def test_switch_spots_is_a_collision(self):
        grid = MapfGrid(['..'])

        agents_starts = ((0, 0), (0, 1),)
        agents_goals = ((0, 1), (0, 0))

        determinstic_env = MapfEnv(grid, 2, agents_starts, agents_goals,
                                   0, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING, OptimizationCriteria.Makespan)

        s, r, done, _ = determinstic_env.step(vector_action_to_integer((RIGHT, LEFT)))

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

        agents_starts = ((1, 2), (2, 1))
        # don't care
        agents_goals = ((0, 0), (2, 3))

        env = MapfEnv(grid, 2, agents_starts, agents_goals,
                      0, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING, OptimizationCriteria.Makespan)

        expected_locations = [
            ((0, 2), (2, 2)),
            ((0, 2), (2, 0)),
            ((0, 2), (1, 1)),
            ((0, 2), (2, 1)),
            ((1, 1), (2, 2)),
            ((1, 1), (2, 0)),
            ((1, 1), (1, 1)),
            ((1, 1), (2, 1)),
            ((1, 3), (2, 2)),
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

        expected_states = [env.locations_to_state(loc) for loc in expected_locations]

        self.assertSetEqual(set(expected_states),
                            set(env.predecessors(env.s)))

    def test_similar_transitions_probability_summed(self):
        grid = MapfGrid(['..',
                         '..'])
        env = MapfEnv(grid, 1, ((0, 0),), ((1, 1),),
                      0.1, REWARD_OF_CLASH, REWARD_OF_GOAL, REWARD_OF_LIVING, OptimizationCriteria.Makespan)

        a = vector_action_to_integer((STAY, STAY))
        self.assertEqual(env.P[env.s][a], [((1, False), env.s, REWARD_OF_LIVING, False)])

    # def test_single_agent_action_transitions(self):
    #     env = create_mapf_env('empty-8-8', 1, 2, 0.2, -1000, -1, -1, OptimizationCriteria.Makespan)
    #
    #     local_action = ACTIONS.index('RIGHT')
    #
    #     eq_joint_action = vector_action_to_integer(('STAY', 'RIGHT'))
    #
    #     self.assertEqual(env.P[env.s][eq_joint_action], env.get_single_agent_transitions(env.s, 1, local_action))

    def test_reward_multiagent_soc(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....'])

        start_locations = ((0, 0), (3, 3), (1, 1))
        goal_locations = ((0, 1), (1, 3), (1, 2))

        determinstic_env = MapfEnv(grid,
                                   3,
                                   start_locations,
                                   goal_locations,
                                   0,
                                   REWARD_OF_CLASH,
                                   REWARD_OF_GOAL,
                                   REWARD_OF_LIVING,
                                   OptimizationCriteria.SoC)
        total_reward = 0
        right_up_right = vector_action_to_integer((RIGHT, UP, RIGHT))
        s, r, done, _ = determinstic_env.step(right_up_right)
        total_reward += r
        self.assertFalse(done)

        stay_up_stay = vector_action_to_integer((STAY, UP, STAY))
        s, r, done, _ = determinstic_env.step(stay_up_stay)
        total_reward += r
        self.assertEqual(s, determinstic_env.locations_to_state(goal_locations))
        self.assertTrue(done)
        self.assertEqual(total_reward, 4 * REWARD_OF_LIVING + REWARD_OF_GOAL)

    def test_reawrd_multiagent_makespan(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....'])

        start_locations = ((0, 0), (3, 3), (1, 1))
        goal_locations = ((0, 1), (1, 3), (1, 2))

        determinstic_env = MapfEnv(grid, 3, start_locations, goal_locations, 0, REWARD_OF_CLASH, REWARD_OF_GOAL,
                                   REWARD_OF_LIVING,
                                   OptimizationCriteria.Makespan)

        total_reward = 0
        right_up_right = vector_action_to_integer((RIGHT, UP, RIGHT))
        s, r, done, _ = determinstic_env.step(right_up_right)
        total_reward += r
        self.assertFalse(done)

        stay_up_stay = vector_action_to_integer((STAY, UP, STAY))
        s, r, done, _ = determinstic_env.step(stay_up_stay)
        total_reward += r
        self.assertEqual(s, determinstic_env.locations_to_state(goal_locations))
        self.assertTrue(done)
        self.assertEqual(total_reward, 2 * REWARD_OF_LIVING + REWARD_OF_GOAL)

    def test_reward_single_agent_soc(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....',
            '....'])

        start_locations = ((0, 0),)
        goal_locations = ((4, 0),)

        determinstic_env = MapfEnv(grid, 1, start_locations, goal_locations, 0, REWARD_OF_CLASH, REWARD_OF_GOAL,
                                   REWARD_OF_LIVING, OptimizationCriteria.SoC)
        total_reward = 0
        down_action = vector_action_to_integer((DOWN,))
        _, r, _, _ = determinstic_env.step(down_action)
        total_reward += r
        _, r, _, _ = determinstic_env.step(down_action)
        total_reward += r
        _, r, _, _ = determinstic_env.step(down_action)
        total_reward += r
        s, r, done, _ = determinstic_env.step(down_action)
        total_reward += r

        self.assertEqual(s, determinstic_env.locations_to_state(goal_locations))
        self.assertEqual(r, REWARD_OF_LIVING + REWARD_OF_GOAL)
        self.assertEqual(total_reward, REWARD_OF_GOAL + 4 * REWARD_OF_LIVING)

    def test_reward_single_agent_makespan(self):
        grid = MapfGrid([
            '....',
            '....',
            '....',
            '....',
            '....'])

        start_locations = ((0, 0),)
        goal_locations = ((4, 0),)

        determinstic_env = MapfEnv(grid, 1, start_locations, goal_locations, 0, REWARD_OF_CLASH, REWARD_OF_GOAL,
                                   REWARD_OF_LIVING, OptimizationCriteria.Makespan)
        total_reward = 0
        down_action = vector_action_to_integer((DOWN,))
        _, r, _, _ = determinstic_env.step(down_action)
        total_reward += r
        _, r, _, _ = determinstic_env.step(down_action)
        total_reward += r
        _, r, _, _ = determinstic_env.step(down_action)
        total_reward += r
        s, r, done, _ = determinstic_env.step(down_action)
        total_reward += r

        self.assertEqual(s, determinstic_env.locations_to_state(goal_locations))
        self.assertEqual(r, REWARD_OF_LIVING + REWARD_OF_GOAL)

        self.assertEqual(total_reward, REWARD_OF_GOAL + 4 * REWARD_OF_LIVING)


if __name__ == '__main__':
    unittest.main(verbosity=2)
