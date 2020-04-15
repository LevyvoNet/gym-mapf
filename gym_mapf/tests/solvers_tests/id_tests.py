import unittest

from gym_mapf.envs.utils import MapfGrid
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    integer_action_to_vector,
                                    UP, DOWN, RIGHT, LEFT, STAY,
                                    ACTIONS)

from gym_mapf.solvers.id import ID
from gym_mapf.solvers.utils import best_joint_policy
from gym_mapf.solvers.vi import value_iteration_planning


class IdTests(unittest.TestCase):
    def test_corridor_switch_indepedent_vs_merged(self):
        grid = MapfGrid(['...',
                         '@.@'])
        agents_starts = ((0, 0), (0, 2))
        agents_goals = ((0, 2), (0, 0))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -1, 1, -0.01)

        independent_joiont_policy = best_joint_policy(env, [[0], [1]], value_iteration_planning)
        merged_joint_policy = best_joint_policy(env, [[0, 1]], value_iteration_planning)

        interesting_state = env.locations_to_state(((1, 1), (0, 1)))

        expected_possible_actions = [vector_action_to_integer((STAY, UP)),
                                     vector_action_to_integer((DOWN, UP))]

        # Assert independent_joint_policy just choose the most efficient action
        self.assertEqual(independent_joiont_policy(interesting_state), vector_action_to_integer((UP, LEFT)))

        # Assert merged_joint_policy avoids collision
        self.assertIn(merged_joint_policy(interesting_state), expected_possible_actions)

    def test_two_columns_independent_vs_merged(self):
        grid = MapfGrid(['..',
                         '..',
                         '..'])
        agents_starts = ((0, 0), (0, 1))
        agents_goals = ((2, 0), (2, 1))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.01, -1, 1, -0.1)

        independent_joiont_policy = best_joint_policy(env, [[0], [1]], value_iteration_planning)
        merged_joint_policy = best_joint_policy(env, [[0, 1]], value_iteration_planning)

        interesting_state = env.locations_to_state(((0, 0), (0, 1)))

        expected_possible_actions = [vector_action_to_integer((LEFT, RIGHT))]

        # Assert independent_joint_policy just choose the most efficient action
        self.assertEqual(independent_joiont_policy(interesting_state), vector_action_to_integer((DOWN, DOWN)))

        # Assert merged_joint_policy avoids collision
        self.assertIn(merged_joint_policy(interesting_state), expected_possible_actions)

    def test_corridor_switch_ID_merge_agents(self):
        grid = MapfGrid(['...',
                         '@.@'])
        agents_starts = ((0, 0), (0, 2))
        agents_goals = ((0, 2), (0, 0))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -1, 1, -0.01)

        policy = ID(env)

        interesting_state = env.locations_to_state(((1, 1), (0, 1)))

        expected_possible_actions = [vector_action_to_integer((STAY, UP)),
                                     vector_action_to_integer((DOWN, UP))]

        self.assertIn(policy(interesting_state), expected_possible_actions)

    def test_narrow_empty_grid(self):
        grid = MapfGrid(['....'])

        agents_starts = ((0, 1), (0, 2))
        agents_goals = ((0, 0), (0, 3))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -1, 1, -0.01)

        joint_policy = ID(env)

        self.assertEqual(joint_policy(env.s), vector_action_to_integer((LEFT, RIGHT)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
