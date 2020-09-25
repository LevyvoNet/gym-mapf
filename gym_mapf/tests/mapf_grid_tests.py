import unittest
import os
from gym_mapf.envs.utils import parse_map_file
from gym_mapf.envs.grid import MapfGrid, EmptyCell, ObstacleCell
from gym_mapf.tests import MAPS_DIR


class MapfGridTest(unittest.TestCase):
    def test_empty_8_8(self):
        map_file_path = os.path.abspath(os.path.join(__file__, MAPS_DIR, 'empty-8-8/empty-8-8.map'))
        grid = MapfGrid(parse_map_file(map_file_path))

        self.assertTrue(grid[0, 0] is EmptyCell)
        self.assertTrue(grid[1, 1] is EmptyCell)
        self.assertTrue(grid[0, 1] is EmptyCell)
        self.assertTrue(grid[2, 1] is EmptyCell)
        self.assertTrue(grid[7, 7] is EmptyCell)

        with self.assertRaises(IndexError):
            grid[8, 1]

    def test_berlin_1_256(self):
        map_file_path = os.path.abspath(os.path.join(__file__, MAPS_DIR, 'Berlin_1_256/Berlin_1_256.map'))
        grid = MapfGrid(parse_map_file(map_file_path))

        self.assertTrue(grid[0, 0] is EmptyCell)
        self.assertTrue(grid[0, 104] is EmptyCell)
        self.assertTrue(grid[0, 105] is ObstacleCell)
        self.assertTrue(grid[0, 106] is ObstacleCell)
        self.assertTrue(grid[0, 107] is ObstacleCell)
        self.assertTrue(grid[0, 108] is ObstacleCell)
        self.assertTrue(grid[0, 109] is EmptyCell)


if __name__ == '__main__':
    unittest.main(verbosity=2)
