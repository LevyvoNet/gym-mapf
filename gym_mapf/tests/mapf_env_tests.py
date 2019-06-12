import unittest
import os

from gym_mapf.utils.executor import UP, DOWN, LEFT, RIGHT
from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.utils.grid import MapfGrid
from gym_mapf.envs import parse_scen_file


class MapfEnvTest(unittest.TestCase):
    def test_transition_function_empty_grid(self):
        map_file_path = os.path.abspath(os.path.join(__file__, '../../maps/empty-8-8/empty-8-8.map'))
        # scen_file_path = os.path.abspath(os.path.join(__file__, '../../maps/empty-8-8/empty-8-8-even-1.scen'))
        n_agents = 2
        grid = MapfGrid(map_file_path)

        # agents are starting a
        agent_starts, agents_goals = [(0, 0), (7, 7)], [(0, 2), (5, 7)]
        env = MapfEnv(grid, agent_starts, agents_goals)

        import ipdb
        ipdb.set_trace()
        self.assertEqual(env.P[env.s][(RIGHT, UP)],
                         [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
