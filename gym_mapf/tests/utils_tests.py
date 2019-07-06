import unittest

from gym_mapf.envs.utils import create_mapf_env
from gym_mapf.envs import integer_to_vector
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
        self.assertEqual(integer_to_vector(10, 4 * 3, 2, lambda n: (int(n / 3), n % 4)),
                         ((3, 2), (0, 0)))
        self.assertEqual(integer_to_vector(13, 4 * 3, 2, lambda n: (int(n / 3), n % 4)),
                         ((0, 1), (0, 1)))
        self.assertEqual(integer_to_vector(14, 4 * 3, 2, lambda n: (int(n / 3), n % 4)),
                         ((0, 2), (0, 1)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
