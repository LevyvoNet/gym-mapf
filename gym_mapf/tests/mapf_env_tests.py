import unittest
import os

from gym_mapf.utils.executor import UP, DOWN, LEFT, RIGHT
from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.utils.grid import MapfGrid
from gym_mapf.utils.state import MapfState
from gym_mapf.envs import parse_scen_file


class MapfEnvTest(unittest.TestCase):
    def test_transition_function_empty_grid(self):
        map_file_path = os.path.abspath(os.path.join(__file__, '../../maps/empty-8-8/empty-8-8.map'))
        # scen_file_path = os.path.abspath(os.path.join(__file__, '../../maps/empty-8-8/empty-8-8-even-1.scen'))
        grid = MapfGrid(map_file_path)

        # agents are starting a
        agent_starts, agents_goals = [(0, 0), (7, 7)], [(0, 2), (5, 7)]
        env = MapfEnv(grid, agent_starts, agents_goals)

        first_step_transitions = [(round(prob, 2), next_state, reward, done)
                                  for (prob, next_state, reward, done) in env.P[env.s][(RIGHT, UP)]]

        self.assertEqual(set(first_step_transitions), {
            (0.64, MapfState(grid, [(0, 1), (6, 7)]), 0.0, False),  # (RIGHT, UP)
            (0.08, MapfState(grid, [(1, 0), (6, 7)]), 0.0, False),  # (DOWN, UP)
            (0.08, MapfState(grid, [(0, 0), (6, 7)]), 0.0, False),  # (UP, UP)
            (0.08, MapfState(grid, [(0, 1), (7, 7)]), 0.0, False),  # (RIGHT, RIGHT)
            (0.08, MapfState(grid, [(0, 1), (7, 6)]), 0.0, False),  # (RIGHT, LEFT)
            (0.01, MapfState(grid, [(1, 0), (7, 7)]), 0.0, False),  # (DOWN, RIGHT)
            (0.01, MapfState(grid, [(1, 0), (7, 6)]), 0.0, False),  # (DOWN, LEFT)
            (0.01, MapfState(grid, [(0, 0), (7, 7)]), 0.0, False),  # (UP, RIGHT)
            (0.01, MapfState(grid, [(0, 0), (7, 6)]), 0.0, False)  # (UP, LEFT)
        })

        wish_state = MapfState(grid, [(0, 1), (6, 7)])
        second_step_transitions = [(round(prob, 2), next_state, reward, done)
                                   for (prob, next_state, reward, done) in env.P[wish_state][(RIGHT, UP)]]

        # [(0,0), (7,7)]
        self.assertEqual(set(second_step_transitions), {
            (0.64, MapfState(grid, [(0, 2), (5, 7)]), 1.0, True),  # (RIGHT, UP)
            (0.08, MapfState(grid, [(1, 1), (5, 7)]), 0.0, False),  # (DOWN, UP)
            (0.08, MapfState(grid, [(0, 1), (5, 7)]), 0.0, False),  # (UP, UP)
            (0.08, MapfState(grid, [(0, 2), (6, 7)]), 0.0, False),  # (RIGHT, RIGHT)
            (0.08, MapfState(grid, [(0, 2), (6, 6)]), 0.0, False),  # (RIGHT, LEFT)
            (0.01, MapfState(grid, [(1, 1), (6, 7)]), 0.0, False),  # (DOWN, RIGHT)
            (0.01, MapfState(grid, [(1, 1), (6, 6)]), 0.0, False),  # (DOWN, LEFT)
            (0.01, MapfState(grid, [(0, 1), (6, 7)]), 0.0, False),  # (UP, RIGHT)
            (0.01, MapfState(grid, [(0, 1), (6, 6)]), 0.0, False)  # (UP, LEFT)
        })


if __name__ == '__main__':
    unittest.main(verbosity=2)
