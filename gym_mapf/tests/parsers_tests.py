import unittest
import os

from gym_mapf.envs.utils import parse_scen_file


class ParsersTest(unittest.TestCase):

    def test_scen_parser_emtpy_8_8(self):
        scen_file_path = os.path.abspath(os.path.join(__file__, '../../maps/empty-8-8/empty-8-8-even-1.scen'))
        agents_starts, agents_goals = parse_scen_file(scen_file_path, 4)

        self.assertEqual(agents_starts, ((0, 0), (5, 3), (1, 7), (0, 5)))
        self.assertEqual(agents_goals, ((1, 0), (5, 6), (6, 4), (7, 4)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
