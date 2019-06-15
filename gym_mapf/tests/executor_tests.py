import unittest
import os

from gym_mapf.utils.grid import MapfGrid
from gym_mapf.utils.state import MapfState
from gym_mapf.utils.executor import UP, DOWN, LEFT, RIGHT, STAY, execute_action
from gym_mapf.envs.utils import parse_map_file


# TODO: Make this test not depend on the file system.
class ExecutorTest(unittest.TestCase):
    def test_moving_on_empty_grid(self):
        map_file_path = os.path.abspath(os.path.join(__file__, '../../maps/empty-8-8/empty-8-8.map'))
        grid = MapfGrid(parse_map_file(map_file_path))

        s = MapfState(grid, [(0, 0), (7, 7)])

        new_state = execute_action(s, (RIGHT, UP))
        self.assertEqual(new_state.agent_locations, [(0, 1), (6, 7)])

        new_state = execute_action(s, (DOWN, LEFT))
        self.assertEqual(new_state.agent_locations, [(1, 0), (7, 6)])

    def test_against_the_wall(self):
        map_file_path = os.path.abspath(os.path.join(__file__, '../../maps/empty-8-8/empty-8-8.map'))
        grid = MapfGrid(parse_map_file(map_file_path))

        s = MapfState(grid, [(0, 0), (7, 7)])

        new_state = execute_action(s, (LEFT, RIGHT))
        self.assertEqual(new_state.agent_locations, [(0, 0), (7, 7)])

    def test_against_obstacle_stays_in_place(self):
        grid = MapfGrid([
            '..@..',
            '..@..',
            '.....',
            '..@..',
            '..@..'])

        s = MapfState(grid, [(0, 1)])  # start near an obstacle.

        new_state = execute_action(s, (RIGHT,))
        self.assertEqual(new_state.agent_locations, [(0, 1)])  # The agent hits an obstacle and should stay in place.

    def test_stay_action(self):
        map_file_path = os.path.abspath(os.path.join(__file__, '../../maps/empty-8-8/empty-8-8.map'))
        grid = MapfGrid(parse_map_file(map_file_path))

        s = MapfState(grid, [(0, 0), (7, 7)])

        new_state = execute_action(s, (STAY, STAY))
        self.assertEqual(new_state.agent_locations, [(0, 0), (7, 7)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
