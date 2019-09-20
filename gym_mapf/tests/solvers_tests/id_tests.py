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


class IdTests(unittest.TestCase):
    def test_corridor_switch_ID(self):
        grid = MapfGrid(['...',
                         '@.@'])
        agents_starts = vector_state_to_integer(grid, ((0, 0), (0, 2)))
        agents_goals = vector_state_to_integer(grid, ((0, 2), (0, 0)))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -1, 10, -0.1)

        policy = ID(env)

        interesting_state = vector_state_to_integer(env.grid, ((1, 1), (0, 1)))
        expected_possible_actions = [vector_action_to_integer((STAY, UP)),
                                     vector_action_to_integer((DOWN, UP))]

        self.assertIn(policy(interesting_state), expected_possible_actions)


if __name__ == '__main__':
    unittest.main(verbosity=2)
