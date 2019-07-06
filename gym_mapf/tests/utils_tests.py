import unittest

from gym_mapf.envs.utils import create_mapf_env
from gym_mapf.envs import integer_to_vector, vector_to_integer
from gym_mapf.envs.mapf_env import ACTIONS, UP, RIGHT, DOWN, LEFT, STAY


class UtilsTest(unittest.TestCase):
    def test_create_mapf_env(self):
        empty_8_8_1 = create_mapf_env(map_name='empty-8-8',
                                      scen_id=1,
                                      n_agents=2,
                                      right_fail=0.1,
                                      left_fail=0.1,
                                      reward_of_clash=-1000.0,
                                      reward_of_goal=100.0,
                                      reward_of_living=0.0)

        self.assertEqual(empty_8_8_1.s, ((0, 0), (5, 3)))

        empty_48_48_16 = create_mapf_env(map_name='empty-48-48',
                                         scen_id=16,
                                         n_agents=2,
                                         right_fail=0.1,
                                         left_fail=0.1,
                                         reward_of_clash=-1000.0,
                                         reward_of_goal=100.0,
                                         reward_of_living=0.0)

        self.assertEqual(empty_48_48_16.s, ((40, 42), (17, 2)))

    def test_integer_to_vector(self):
        # state in a 4x4 grid for a single agent.
        self.assertEqual(integer_to_vector(10, 4, 2, lambda n: n), (2, 2))

        # action for 3 agents
        self.assertEqual(integer_to_vector(28, 5, 3, lambda n: ACTIONS[n]), (LEFT, UP, RIGHT))

        # state in a 4x3 grid for two agents.
        self.assertEqual(integer_to_vector(10, 4 * 3, 2, lambda n: (int(n / 3), n % 3)),
                         ((3, 1), (0, 0)))
        self.assertEqual(integer_to_vector(13, 4 * 3, 2, lambda n: (int(n / 3), n % 3)),
                         ((0, 1), (0, 1)))
        self.assertEqual(integer_to_vector(14, 4 * 3, 2, lambda n: (int(n / 3), n % 3)),
                         ((0, 2), (0, 1)))
        self.assertEqual(integer_to_vector(23, 4 * 3, 2, lambda n: (int(n / 3), n % 3)),
                         ((3, 2), (0, 1)))
        self.assertEqual(integer_to_vector(143, 4 * 3, 2, lambda n: (int(n / 3), n % 3)),
                         ((3, 2), (3, 2)))

    def test_vector_to_integer(self):
        # state in a 4x4 grid for a single agent.
        self.assertEqual(vector_to_integer((2, 2), 4, lambda n: n), 10)

        # action for 3 agents
        self.assertEqual(vector_to_integer((LEFT, UP, RIGHT), 5, lambda a: ACTIONS.index(a)), 28)

        # state in a 4x3 grid for two agents.
        self.assertEqual(vector_to_integer(((3, 1), (0, 0)), 4 * 3, lambda v: 3 * v[0] + v[1]), 10)
        self.assertEqual(vector_to_integer(((0, 1), (0, 1)), 4 * 3, lambda v: 3 * v[0] + v[1]), 13)
        self.assertEqual(vector_to_integer(((0, 2), (0, 1)), 4 * 3, lambda v: 3 * v[0] + v[1]), 14)
        self.assertEqual(vector_to_integer(((3, 2), (0, 1)), 4 * 3, lambda v: 3 * v[0] + v[1]), 23)
        self.assertEqual(vector_to_integer(((3, 2), (3, 2)), 4 * 3, lambda v: 3 * v[0] + v[1]), 143)


if __name__ == '__main__':
    unittest.main(verbosity=2)
