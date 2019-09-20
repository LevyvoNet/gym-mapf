import unittest

from gym_mapf.solvers.utils import cross_policies, detect_conflict
from gym_mapf.envs.utils import MapfGrid, get_local_view
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    vector_state_to_integer,
                                    integer_state_to_vector,
                                    integer_action_to_vector,
                                    UP, DOWN, RIGHT, LEFT, STAY,
                                    ACTIONS)

class SolversUtilsTests(unittest.TestCase):
    def test_detect_conflict_finds_classical_conflict(self):
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

        joint_policy = cross_policies([policy1.get, policy2.get], [get_local_view(env, [0]), get_local_view(env, [1])])

        self.assertEqual(detect_conflict(env, joint_policy),
                         (0,
                          vector_state_to_integer(env.grid, ((0, 0),)),
                          1,
                          vector_state_to_integer(env.grid, ((0, 2),)),
                          vector_state_to_integer(env.grid, ((0, 1),)),
                          )
                         )

    def test_detect_conflict_return_none_when_no_conflict(self):
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

        joint_policy = cross_policies([policy1.get, policy2.get], [get_local_view(env, [0]), get_local_view(env, [1])])

        self.assertEqual(detect_conflict(env, joint_policy), None)
