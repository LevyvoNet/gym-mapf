import unittest
import os
from gym_mapf.envs.utils import parse_map_file
from gym_mapf.envs.grid import (MapfGrid,
                                EmptyCell,
                                ObstacleCell,
                                SingleAgentState,
                                SingleAgentAction,
                                vector_to_integer_multiple_numbers,
                                integer_to_vector_multiple_numbers,
                                integer_to_vector,
                                vector_to_integer,
                                integer_action_to_vector,
                                vector_action_to_integer)
from gym_mapf.envs.mapf_env import MultiAgentAction, MultiAgentState, MultiAgentStateSpace, MultiAgentActionSpace
from gym_mapf.tests import MAPS_DIR


class MapfGridTest(unittest.TestCase):
    def test_empty_8_8(self):
        map_file_path = os.path.abspath(os.path.join(__file__, MAPS_DIR, 'empty-8-8/empty-8-8.map'))
        grid = MapfGrid(parse_map_file(map_file_path))

        self.assertTrue(grid[SingleAgentState(0, 0)] is EmptyCell)
        self.assertTrue(grid[SingleAgentState(1, 1)] is EmptyCell)
        self.assertTrue(grid[SingleAgentState(0, 1)] is EmptyCell)
        self.assertTrue(grid[SingleAgentState(2, 1)] is EmptyCell)
        self.assertTrue(grid[SingleAgentState(7, 7)] is EmptyCell)

        with self.assertRaises(IndexError):
            grid[SingleAgentState(8, 1)]

    def test_berlin_1_256(self):
        map_file_path = os.path.abspath(os.path.join(__file__, MAPS_DIR, 'Berlin_1_256/Berlin_1_256.map'))
        grid = MapfGrid(parse_map_file(map_file_path))

        self.assertTrue(grid[SingleAgentState(0, 0)] is EmptyCell)
        self.assertTrue(grid[SingleAgentState(0, 104)] is EmptyCell)
        self.assertTrue(grid[SingleAgentState(0, 105)] is ObstacleCell)
        self.assertTrue(grid[SingleAgentState(0, 106)] is ObstacleCell)
        self.assertTrue(grid[SingleAgentState(0, 107)] is ObstacleCell)
        self.assertTrue(grid[SingleAgentState(0, 108)] is ObstacleCell)
        self.assertTrue(grid[SingleAgentState(0, 109)] is EmptyCell)

    def test_action_space_naive(self):
        action_space = MultiAgentActionSpace([0, 1])
        expected_actions = [
            MultiAgentAction({0: SingleAgentAction.UP, 1: SingleAgentAction.UP}),
            MultiAgentAction({0: SingleAgentAction.UP, 1: SingleAgentAction.RIGHT}),
            MultiAgentAction({0: SingleAgentAction.UP, 1: SingleAgentAction.DOWN}),
            MultiAgentAction({0: SingleAgentAction.UP, 1: SingleAgentAction.LEFT}),
            MultiAgentAction({0: SingleAgentAction.UP, 1: SingleAgentAction.STAY}),

            MultiAgentAction({0: SingleAgentAction.RIGHT, 1: SingleAgentAction.UP}),
            MultiAgentAction({0: SingleAgentAction.RIGHT, 1: SingleAgentAction.RIGHT}),
            MultiAgentAction({0: SingleAgentAction.RIGHT, 1: SingleAgentAction.DOWN}),
            MultiAgentAction({0: SingleAgentAction.RIGHT, 1: SingleAgentAction.LEFT}),
            MultiAgentAction({0: SingleAgentAction.RIGHT, 1: SingleAgentAction.STAY}),

            MultiAgentAction({0: SingleAgentAction.DOWN, 1: SingleAgentAction.UP}),
            MultiAgentAction({0: SingleAgentAction.DOWN, 1: SingleAgentAction.RIGHT}),
            MultiAgentAction({0: SingleAgentAction.DOWN, 1: SingleAgentAction.DOWN}),
            MultiAgentAction({0: SingleAgentAction.DOWN, 1: SingleAgentAction.LEFT}),
            MultiAgentAction({0: SingleAgentAction.DOWN, 1: SingleAgentAction.STAY}),

            MultiAgentAction({0: SingleAgentAction.LEFT, 1: SingleAgentAction.UP}),
            MultiAgentAction({0: SingleAgentAction.LEFT, 1: SingleAgentAction.RIGHT}),
            MultiAgentAction({0: SingleAgentAction.LEFT, 1: SingleAgentAction.DOWN}),
            MultiAgentAction({0: SingleAgentAction.LEFT, 1: SingleAgentAction.LEFT}),
            MultiAgentAction({0: SingleAgentAction.LEFT, 1: SingleAgentAction.STAY}),

            MultiAgentAction({0: SingleAgentAction.STAY, 1: SingleAgentAction.UP}),
            MultiAgentAction({0: SingleAgentAction.STAY, 1: SingleAgentAction.RIGHT}),
            MultiAgentAction({0: SingleAgentAction.STAY, 1: SingleAgentAction.DOWN}),
            MultiAgentAction({0: SingleAgentAction.STAY, 1: SingleAgentAction.LEFT}),
            MultiAgentAction({0: SingleAgentAction.STAY, 1: SingleAgentAction.STAY}),
        ]

        self.assertSetEqual(set(expected_actions), set(action_space))

    def test_action_space_skip_agent(self):
        action_space = MultiAgentActionSpace([0, 2])
        expected_actions = [
            MultiAgentAction({0: SingleAgentAction.UP, 2: SingleAgentAction.UP}),
            MultiAgentAction({0: SingleAgentAction.UP, 2: SingleAgentAction.RIGHT}),
            MultiAgentAction({0: SingleAgentAction.UP, 2: SingleAgentAction.DOWN}),
            MultiAgentAction({0: SingleAgentAction.UP, 2: SingleAgentAction.LEFT}),
            MultiAgentAction({0: SingleAgentAction.UP, 2: SingleAgentAction.STAY}),

            MultiAgentAction({0: SingleAgentAction.RIGHT, 2: SingleAgentAction.UP}),
            MultiAgentAction({0: SingleAgentAction.RIGHT, 2: SingleAgentAction.RIGHT}),
            MultiAgentAction({0: SingleAgentAction.RIGHT, 2: SingleAgentAction.DOWN}),
            MultiAgentAction({0: SingleAgentAction.RIGHT, 2: SingleAgentAction.LEFT}),
            MultiAgentAction({0: SingleAgentAction.RIGHT, 2: SingleAgentAction.STAY}),

            MultiAgentAction({0: SingleAgentAction.DOWN, 2: SingleAgentAction.UP}),
            MultiAgentAction({0: SingleAgentAction.DOWN, 2: SingleAgentAction.RIGHT}),
            MultiAgentAction({0: SingleAgentAction.DOWN, 2: SingleAgentAction.DOWN}),
            MultiAgentAction({0: SingleAgentAction.DOWN, 2: SingleAgentAction.LEFT}),
            MultiAgentAction({0: SingleAgentAction.DOWN, 2: SingleAgentAction.STAY}),

            MultiAgentAction({0: SingleAgentAction.LEFT, 2: SingleAgentAction.UP}),
            MultiAgentAction({0: SingleAgentAction.LEFT, 2: SingleAgentAction.RIGHT}),
            MultiAgentAction({0: SingleAgentAction.LEFT, 2: SingleAgentAction.DOWN}),
            MultiAgentAction({0: SingleAgentAction.LEFT, 2: SingleAgentAction.LEFT}),
            MultiAgentAction({0: SingleAgentAction.LEFT, 2: SingleAgentAction.STAY}),

            MultiAgentAction({0: SingleAgentAction.STAY, 2: SingleAgentAction.UP}),
            MultiAgentAction({0: SingleAgentAction.STAY, 2: SingleAgentAction.RIGHT}),
            MultiAgentAction({0: SingleAgentAction.STAY, 2: SingleAgentAction.DOWN}),
            MultiAgentAction({0: SingleAgentAction.STAY, 2: SingleAgentAction.LEFT}),
            MultiAgentAction({0: SingleAgentAction.STAY, 2: SingleAgentAction.STAY}),
        ]

        self.assertSetEqual(set(expected_actions), set(action_space))

    def test_state_space_naive(self):
        grid_lines = [
            '...',
            '.@.',
            '...'
        ]
        grid = MapfGrid(grid_lines)

        state_space = MultiAgentStateSpace([0, 1], grid)

        expected_states = [
            MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(2, 2)}, grid),

            MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(2, 2)}, grid),

            MultiAgentState({0: SingleAgentState(0, 2), 1: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(0, 2), 1: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(0, 2), 1: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(0, 2), 1: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(0, 2), 1: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(0, 2), 1: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(0, 2), 1: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(0, 2), 1: SingleAgentState(2, 2)}, grid),

            MultiAgentState({0: SingleAgentState(1, 0), 1: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(1, 0), 1: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(1, 0), 1: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(1, 0), 1: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(1, 0), 1: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(1, 0), 1: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(1, 0), 1: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(1, 0), 1: SingleAgentState(2, 2)}, grid),

            MultiAgentState({0: SingleAgentState(1, 2), 1: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(1, 2), 1: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(1, 2), 1: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(1, 2), 1: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(1, 2), 1: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(1, 2), 1: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(1, 2), 1: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(1, 2), 1: SingleAgentState(2, 2)}, grid),

            MultiAgentState({0: SingleAgentState(2, 0), 1: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(2, 0), 1: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(2, 0), 1: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(2, 0), 1: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(2, 0), 1: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(2, 0), 1: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(2, 0), 1: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(2, 0), 1: SingleAgentState(2, 2)}, grid),

            MultiAgentState({0: SingleAgentState(2, 1), 1: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(2, 1), 1: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(2, 1), 1: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(2, 1), 1: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(2, 1), 1: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(2, 1), 1: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(2, 1), 1: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(2, 1), 1: SingleAgentState(2, 2)}, grid),

            MultiAgentState({0: SingleAgentState(2, 2), 1: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(2, 2), 1: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(2, 2), 1: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(2, 2), 1: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(2, 2), 1: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(2, 2), 1: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(2, 2), 1: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(2, 2), 1: SingleAgentState(2, 2)}, grid),

        ]

        self.assertSetEqual(set(expected_states), set(state_space))

    def test_state_space_agent_skip(self):
        grid_lines = [
            '...',
            '.@.',
            '...'
        ]

        grid = MapfGrid(grid_lines)
        state_space = MultiAgentStateSpace([0, 2], grid)

        expected_states = [
            MultiAgentState({0: SingleAgentState(0, 0), 2: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(0, 0), 2: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(0, 0), 2: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(0, 0), 2: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(0, 0), 2: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(0, 0), 2: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(0, 0), 2: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(0, 0), 2: SingleAgentState(2, 2)}, grid),

            MultiAgentState({0: SingleAgentState(0, 1), 2: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(0, 1), 2: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(0, 1), 2: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(0, 1), 2: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(0, 1), 2: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(0, 1), 2: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(0, 1), 2: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(0, 1), 2: SingleAgentState(2, 2)}, grid),

            MultiAgentState({0: SingleAgentState(0, 2), 2: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(0, 2), 2: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(0, 2), 2: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(0, 2), 2: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(0, 2), 2: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(0, 2), 2: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(0, 2), 2: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(0, 2), 2: SingleAgentState(2, 2)}, grid),

            MultiAgentState({0: SingleAgentState(1, 0), 2: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(1, 0), 2: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(1, 0), 2: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(1, 0), 2: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(1, 0), 2: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(1, 0), 2: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(1, 0), 2: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(1, 0), 2: SingleAgentState(2, 2)}, grid),

            MultiAgentState({0: SingleAgentState(1, 2), 2: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(1, 2), 2: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(1, 2), 2: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(1, 2), 2: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(1, 2), 2: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(1, 2), 2: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(1, 2), 2: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(1, 2), 2: SingleAgentState(2, 2)}, grid),

            MultiAgentState({0: SingleAgentState(2, 0), 2: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(2, 0), 2: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(2, 0), 2: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(2, 0), 2: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(2, 0), 2: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(2, 0), 2: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(2, 0), 2: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(2, 0), 2: SingleAgentState(2, 2)}, grid),

            MultiAgentState({0: SingleAgentState(2, 1), 2: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(2, 1), 2: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(2, 1), 2: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(2, 1), 2: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(2, 1), 2: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(2, 1), 2: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(2, 1), 2: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(2, 1), 2: SingleAgentState(2, 2)}, grid),

            MultiAgentState({0: SingleAgentState(2, 2), 2: SingleAgentState(0, 0)}, grid),
            MultiAgentState({0: SingleAgentState(2, 2), 2: SingleAgentState(0, 1)}, grid),
            MultiAgentState({0: SingleAgentState(2, 2), 2: SingleAgentState(0, 2)}, grid),
            MultiAgentState({0: SingleAgentState(2, 2), 2: SingleAgentState(1, 0)}, grid),
            # NOTE: (1,2) state is missing here because it is not a legal state (has an obstacle)
            MultiAgentState({0: SingleAgentState(2, 2), 2: SingleAgentState(1, 2)}, grid),
            MultiAgentState({0: SingleAgentState(2, 2), 2: SingleAgentState(2, 0)}, grid),
            MultiAgentState({0: SingleAgentState(2, 2), 2: SingleAgentState(2, 1)}, grid),
            MultiAgentState({0: SingleAgentState(2, 2), 2: SingleAgentState(2, 2)}, grid),

        ]

        self.assertSetEqual(set(expected_states), set(state_space))

    def test_integer_to_vector(self):
        # state in a 4x4 grid for a single agent.
        self.assertEqual(integer_to_vector(10, [4] * 2, 2, lambda n: n), (2, 2))

        val_to_action = {a.value:a for a in SingleAgentAction}
        # action for 3 agents
        self.assertEqual(integer_to_vector(28, [len(SingleAgentAction)] * 3, 3, lambda n: val_to_action[n]),
                         (SingleAgentAction.DOWN, SingleAgentAction.STAY, SingleAgentAction.UP))

        # state in a 4x3 grid for two agents.
        self.assertEqual(integer_to_vector(10, [4 * 3] * 2, 2, lambda n: (int(n / 3), n % 3)),
                         ((3, 1), (0, 0)))
        self.assertEqual(integer_to_vector(13, [4 * 3] * 2, 2, lambda n: (int(n / 3), n % 3)),
                         ((0, 1), (0, 1)))
        self.assertEqual(integer_to_vector(14, [4 * 3] * 2, 2, lambda n: (int(n / 3), n % 3)),
                         ((0, 2), (0, 1)))
        self.assertEqual(integer_to_vector(23, [4 * 3] * 2, 2, lambda n: (int(n / 3), n % 3)),
                         ((3, 2), (0, 1)))
        self.assertEqual(integer_to_vector(143, [4 * 3] * 2, 2, lambda n: (int(n / 3), n % 3)),
                         ((3, 2), (3, 2)))

    def test_vector_to_integer(self):
        # state in a 4x4 grid for a single agent.
        # XXXX
        # XXXX
        # XVXX
        # XXXX
        self.assertEqual(vector_to_integer((2, 1), [4, 4], lambda n: n), 6)

        # action for 3 agents
        self.assertEqual(vector_to_integer((SingleAgentAction.DOWN, SingleAgentAction.STAY, SingleAgentAction.UP),
                                           [len(SingleAgentAction)] * 3, lambda a: a.value), 28)

        # state in a 4x3 grid for two agents.
        self.assertEqual(vector_to_integer(((3, 1), (0, 0)), [4 * 3] * 2, lambda v: 3 * v[0] + v[1]), 10)
        self.assertEqual(vector_to_integer(((0, 1), (0, 1)), [4 * 3] * 2, lambda v: 3 * v[0] + v[1]), 13)
        self.assertEqual(vector_to_integer(((0, 2), (0, 1)), [4 * 3] * 2, lambda v: 3 * v[0] + v[1]), 14)
        self.assertEqual(vector_to_integer(((3, 2), (0, 1)), [4 * 3] * 2, lambda v: 3 * v[0] + v[1]), 23)
        self.assertEqual(vector_to_integer(((3, 2), (3, 2)), [4 * 3] * 2, lambda v: 3 * v[0] + v[1]), 143)

    def test_vector_to_integer_multiple_option_counts(self):
        self.assertEqual(4, vector_to_integer_multiple_numbers((0, 2), [2, 3], lambda x: x))

    def test_integer_to_vector_multiple_option_counts(self):
        self.assertEqual((0, 2), integer_to_vector_multiple_numbers(4, [2, 3], 2, lambda x: x))

    def test_vector_action_to_integer(self):
        self.assertEqual((SingleAgentAction.DOWN, SingleAgentAction.UP),
                         integer_action_to_vector(
                             vector_action_to_integer((SingleAgentAction.DOWN, SingleAgentAction.UP)), 2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
