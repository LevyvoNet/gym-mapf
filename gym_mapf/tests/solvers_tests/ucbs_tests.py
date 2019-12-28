import unittest

from gym_mapf.solvers.ucbs import best_joint_policy_under_constraint
from gym_mapf.solvers.value_iteration_agent import plan_with_value_iteration
from gym_mapf.envs.utils import MapfGrid, get_local_view
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    integer_action_to_vector,
                                    UP, DOWN, RIGHT, LEFT, STAY,
                                    ACTIONS)


class UcbsTests(unittest.TestCase):
    def test_find_best_policies_with_no_constraints(self):
        grid = MapfGrid(['...',
                         '@.@',
                         '...'])

        agents_starts = ((0, 0), (0, 2))
        agents_goals = ((2, 0), (2, 2))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0, 0, -1, 1, -0.01)

        sum_exptected_reward, joint_policy = best_joint_policy_under_constraint(env,
                                                                                [[], []],
                                                                                plan_with_value_iteration)

        self.assertEqual(joint_policy(env.locations_to_state(agents_starts)), vector_action_to_integer((RIGHT, LEFT)))

    def test_find_best_policies_with_one_constraint(self):
        """Test that adding a constraint changes the selected policies."""
        grid = MapfGrid(['...',
                         '@.@',
                         '...'])

        agents_starts = ((0, 0), (0, 2))
        agents_goals = ((2, 0), (2, 2))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -1, 1, -0.01)

        # This is just for generating local states for the constraint
        local_env = MapfEnv(grid, 1, ((0, 0),), ((2, 0),), 0.1, 0.1, -1, 1, -0.01)

        constraint = (0,
                      local_env.locations_to_state(((0, 0),)),
                      1,
                      local_env.locations_to_state(((0, 2),)),
                      local_env.locations_to_state(((0, 1),)),
                      )
        sum_exptected_reward, joint_policy = best_joint_policy_under_constraint(env,
                                                                                [[constraint], []],
                                                                                plan_with_value_iteration)

        best_action = joint_policy(env.locations_to_state(agents_starts))

        # agent 0 must take an action which has no chance to get him to position (0,1).
        # that means he must not move to the right, and also up or down (because up and down might
        # turn out to be right).
        self.assertIn(best_action, [vector_action_to_integer((STAY, LEFT)),
                                    vector_action_to_integer((LEFT, LEFT))])


if __name__== '__main__':
    unittest.main(verbosity=2)