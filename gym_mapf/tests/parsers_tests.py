import unittest
import os

from gym_mapf.envs import parse_scen_file
from gym_mapf.utils.grid import MapfGrid, EmptyCell, ObstacleCell


class ParsersTest(unittest.TestCase):
    def test_map_parser_empty_8_8(self):
        map_file_path = os.path.abspath(os.path.join(__file__, '../../maps/empty-8-8/empty-8-8.map'))
        grid = MapfGrid(map_file_path)

        self.assertTrue(grid[0, 0] is EmptyCell)
        self.assertTrue(grid[1, 1] is EmptyCell)
        self.assertTrue(grid[0, 1] is EmptyCell)
        self.assertTrue(grid[2, 1] is EmptyCell)
        self.assertTrue(grid[7, 7] is EmptyCell)

        with self.assertRaises(IndexError):
            grid[8, 1]

    def test_map_parser_berlin_1_256(self):
        map_file_path = os.path.abspath(os.path.join(__file__, '../../maps/Berlin_1_256/Berlin_1_256.map'))
        grid = MapfGrid(map_file_path)

        self.assertTrue(grid[0, 0] is EmptyCell)
        self.assertTrue(grid[0, 104] is EmptyCell)
        self.assertTrue(grid[0, 105] is ObstacleCell)
        self.assertTrue(grid[0, 106] is ObstacleCell)
        self.assertTrue(grid[0, 107] is ObstacleCell)
        self.assertTrue(grid[0, 108] is ObstacleCell)
        self.assertTrue(grid[0, 109] is EmptyCell)

    def test_scen_parser_emtpy_8_8(self):
        scen_file_path = os.path.abspath(os.path.join(__file__, '../../maps/empty-8-8/empty-8-8-even-1.scen'))
        agents_starts, agents_goals = parse_scen_file(scen_file_path, 4)

        self.assertEqual(agents_starts, [(0, 0), (5, 3), (1, 7), (0, 5)])
        self.assertEqual(agents_goals, [(1, 0), (5, 6), (6, 4), (7, 4)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
