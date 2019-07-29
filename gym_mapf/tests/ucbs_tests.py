import unittest

from gym_mapf.solvers.ucbs import find_conflict, best_joint_policy_under_constraint, cross_policies
from gym_mapf.envs.utils import MapfGrid
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    vector_state_to_integer,
                                    integer_state_to_vector,
                                    integer_action_to_vector,
                                    UP, DOWN, RIGHT, LEFT, STAY,
                                    ACTIONS)


class UcbsTests(unittest.TestCase):
    def test_find_conflict_finds_classical_conflict(self):
        grid = MapfGrid(['...',
                         '@.@',
                         '...'])

        agents_starts = vector_state_to_integer(grid, ((0, 0), (0, 2)))
        agents_goals = vector_state_to_integer(grid, ((2, 0), (2, 2)))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0, 0, -1, 1, -0.01)

        policy1 = {
            0: ACTIONS.index(RIGHT),
            1: ACTIONS.index(DOWN),
            2: ACTIONS.index(LEFT),
            3: ACTIONS.index(RIGHT),
            4: ACTIONS.index(DOWN),
            5: ACTIONS.index(LEFT),
            6: ACTIONS.index(STAY),
            7: ACTIONS.index(LEFT),
            8: ACTIONS.index(LEFT),
        }

        policy2 = {
            0: ACTIONS.index(RIGHT),
            1: ACTIONS.index(DOWN),
            2: ACTIONS.index(LEFT),
            3: ACTIONS.index(RIGHT),
            4: ACTIONS.index(DOWN),
            5: ACTIONS.index(LEFT),
            6: ACTIONS.index(RIGHT),
            7: ACTIONS.index(RIGHT),
            8: ACTIONS.index(STAY),
        }

        self.assertEqual(find_conflict(env, cross_policies([policy1, policy2], env), 2),
                         (0,
                          vector_state_to_integer(env.grid, ((0, 0),)),
                          1,
                          vector_state_to_integer(env.grid, ((0, 2),))
                          )
                         )

    def test_find_conflict_return_none_when_no_conflict(self):
        grid = MapfGrid(['...',
                         '...',
                         '...'])

        agents_starts = vector_state_to_integer(grid, ((0, 0), (0, 2)))
        agents_goals = vector_state_to_integer(grid, ((2, 0), (2, 2)))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0, 0, -1, 1, -0.01)

        policy1 = {
            0: ACTIONS.index(DOWN),
            1: ACTIONS.index(DOWN),
            2: ACTIONS.index(DOWN),
            3: ACTIONS.index(DOWN),
            4: ACTIONS.index(DOWN),
            5: ACTIONS.index(DOWN),
            6: ACTIONS.index(DOWN),
            7: ACTIONS.index(DOWN),
            8: ACTIONS.index(DOWN),
        }

        policy2 = {
            0: ACTIONS.index(DOWN),
            1: ACTIONS.index(DOWN),
            2: ACTIONS.index(DOWN),
            3: ACTIONS.index(DOWN),
            4: ACTIONS.index(DOWN),
            5: ACTIONS.index(DOWN),
            6: ACTIONS.index(DOWN),
            7: ACTIONS.index(DOWN),
            8: ACTIONS.index(DOWN),
        }

        self.assertEqual(find_conflict(env, cross_policies([policy1, policy2], env), 2),
                         None)

    # def test_find_best_policies_with_constraint(self):
    #     grid = MapfGrid(['...',
    #                      '...',
    #                      '...'])
    #
    #     agents_starts = vector_state_to_integer(grid, ((0, 0), (0, 2)))
    #     agents_goals = vector_state_to_integer(grid, ((2, 0), (2, 2)))
    #
    #     env = MapfEnv(grid, 2, agents_starts, agents_goals, 0, 0, -1, 1, -0.01)
    #
    #     policies = best_joint_policy_under_constraint(env, [], 2)
    #
    #     self.assertEqual(policies, None)
