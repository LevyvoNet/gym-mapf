import unittest
from math import inf

from gym_mapf.envs.utils import MapfGrid
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    vector_state_to_integer,
                                    integer_state_to_vector,
                                    integer_action_to_vector,
                                    UP, DOWN, RIGHT, LEFT, STAY,
                                    ACTIONS)

from gym_mapf.solvers.id import ID


# class IdTests(unittest.TestCase):
    # def test_id_with_basic_conflict(self):
    #     grid = MapfGrid(['.....',
    #                      '@.@@@', ])
    #
    #     agents_starts = vector_state_to_integer(grid, ((0, 0), (0, 4)))
    #     agents_goals = vector_state_to_integer(grid, ((0, 4), (0, 0)))
    #
    #     env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -100, 0, -0.1)
    #
    #     p = ID(env)


if __name__ == '__main__':
    unittest.main(verbosity=2)
