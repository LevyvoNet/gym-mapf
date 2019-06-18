import unittest
from gym_mapf.envs.utils import create_mapf_env


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


if __name__ == '__main__':
    unittest.main(verbosity=2)
