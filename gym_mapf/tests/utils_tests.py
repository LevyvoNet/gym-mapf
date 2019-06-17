import unittest
from gym_mapf.envs.utils import create_mapf_env


class UtilsTest(unittest.TestCase):
    def test_create_mapf_env(self):
        env = create_mapf_env(map_name='empty-8-8',
                              n_agents=2,
                              right_fail=0.1,
                              left_fail=0.1,
                              reward_of_clash=-1000.0,
                              reward_of_goal=100.0,
                              reward_of_living=0.0)

        self.assertEqual(env.s, ((0, 0), (5, 3)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
