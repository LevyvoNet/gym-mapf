import unittest
import os

from gym_mapf.envs.grid import (MapfGrid,
                                SingleAgentState,
                                SingleAgentAction)
from gym_mapf.envs.utils import parse_map_file
from gym_mapf.envs.mapf_env import MapfEnv, OptimizationCriteria, MultiAgentAction, MultiAgentState
from gym_mapf.tests import MAPS_DIR


# TODO: Make this test not depend on the file system.
class ExecutorTest(unittest.TestCase):
    def test_moving_on_empty_grid(self):
        map_file_path = os.path.abspath(os.path.join(__file__, MAPS_DIR, 'empty-8-8/empty-8-8.map'))
        grid = MapfGrid(parse_map_file(map_file_path))

        start_state = MultiAgentState({
            0: SingleAgentState(0, 0),
            1: SingleAgentState(7, 7)
        }, grid)

        goal_state = MultiAgentState({
            1: SingleAgentState(0, 0),
            0: SingleAgentState(7, 7)
        }, grid)

        env = MapfEnv(grid, 2, start_state, goal_state, 0, 0, 0, 0, OptimizationCriteria.Makespan)
        new_state, _, _, _ = env.step(MultiAgentAction({
            0: SingleAgentAction.RIGHT,
            1: SingleAgentAction.UP
        }))

        self.assertEqual(new_state,
                         MultiAgentState({
                             0: SingleAgentState(0, 1),
                             1: SingleAgentState(6, 7)
                         }, grid))

        env.reset()
        new_state, _, _, _ = env.step(MultiAgentAction({
            0: SingleAgentAction.DOWN,
            1: SingleAgentAction.LEFT
        }))
        self.assertEqual(new_state,
                         MultiAgentState({
                             0: SingleAgentState(1, 0),
                             1: SingleAgentState(7, 6)
                         }, grid))

    def test_against_the_wall(self):
        map_file_path = os.path.abspath(os.path.join(__file__, MAPS_DIR, 'empty-8-8/empty-8-8.map'))
        grid = MapfGrid(parse_map_file(map_file_path))

        start_state = MultiAgentState({
            0: SingleAgentState(0, 0),
            1: SingleAgentState(7, 7)
        }, grid)

        goal_state = MultiAgentState({
            1: SingleAgentState(0, 0),
            0: SingleAgentState(7, 7)
        }, grid)

        env = MapfEnv(grid, 2, start_state, goal_state, 0, 0, 0, 0, OptimizationCriteria.Makespan)

        new_state, _, _, _ = env.step(MultiAgentAction({
            0: SingleAgentAction.LEFT,
            1: SingleAgentAction.RIGHT
        }))
        self.assertEqual(new_state, start_state)

    def test_against_obstacle_stays_in_place(self):
        grid = MapfGrid([
            '..@..',
            '..@..',
            '.....',
            '..@..',
            '..@..'])

        # start near an obstacle.
        start_state = MultiAgentState({
            0: SingleAgentState(0, 1),
        }, grid)

        # don't really care
        goal_state = MultiAgentState({
            0: SingleAgentState(1, 0),
        }, grid)

        env = MapfEnv(grid, 2, start_state, goal_state, 0, 0, 0, 0, OptimizationCriteria.Makespan)
        new_state, _, _, _ = env.step(MultiAgentAction({
            0: SingleAgentAction.RIGHT
        }))

        self.assertEqual(new_state, start_state)  # The agent hits an obstacle and should stay in place.

    def test_stay_action(self):
        map_file_path = os.path.abspath(os.path.join(__file__, MAPS_DIR, 'empty-8-8/empty-8-8.map'))
        grid = MapfGrid(parse_map_file(map_file_path))

        start_state = MultiAgentState({
            0: SingleAgentState(0, 0),
            1: SingleAgentState(7, 7)
        }, grid)

        goal_state = MultiAgentState({
            1: SingleAgentState(0, 0),
            0: SingleAgentState(7, 7)
        }, grid)

        env = MapfEnv(grid, 2, start_state, goal_state, 0, 0, 0, 0, OptimizationCriteria.Makespan)

        new_state, _, _, _ = env.step(MultiAgentAction({
            0: SingleAgentAction.STAY,
            1: SingleAgentAction.STAY
        }))
        self.assertEqual(new_state, start_state)


if __name__ == '__main__':
    unittest.main(verbosity=2)
