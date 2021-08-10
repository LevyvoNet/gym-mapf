import unittest

from gym_mapf.envs.utils import create_mapf_env
from gym_mapf.envs.grid import SingleAgentAction, SingleAgentState
from gym_mapf.envs.mapf_env import OptimizationCriteria, MultiAgentAction, MultiAgentState


class UtilsTest(unittest.TestCase):
    def test_create_mapf_env(self):
        empty_8_8_1 = create_mapf_env(map_name='empty-8-8',
                                      scen_id=1,
                                      n_agents=2,
                                      fail_prob=0.2,
                                      reward_of_collision=-1000.0,
                                      reward_of_goal=100.0,
                                      reward_of_living=0.0,
                                      optimization_criteria=OptimizationCriteria.Makespan)

        expected_start_state = MultiAgentState({
            0: SingleAgentState(0, 0),
            1: SingleAgentState(5, 3)
        }, empty_8_8_1.grid)
        self.assertEqual(empty_8_8_1.s, expected_start_state)

        empty_48_48_16 = create_mapf_env(map_name='empty-48-48',
                                         scen_id=16,
                                         n_agents=2,
                                         fail_prob=0.2,
                                         reward_of_collision=-1000.0,
                                         reward_of_goal=100.0,
                                         reward_of_living=0.0,
                                         optimization_criteria=OptimizationCriteria.Makespan)

        expected_start_state = MultiAgentState({
            0: SingleAgentState(40, 42),
            1: SingleAgentState(17, 2)
        }, empty_48_48_16.grid)

        self.assertEqual(empty_48_48_16.s, expected_start_state)


if __name__ == '__main__':
    unittest.main(verbosity=2)
