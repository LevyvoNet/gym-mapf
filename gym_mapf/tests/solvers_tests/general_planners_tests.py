import unittest
from abc import ABCMeta, abstractmethod

from gym_mapf.envs.utils import MapfGrid
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    UP, DOWN, STAY)
from gym_mapf.solvers.utils import Planner
from gym_mapf.solvers import (ValueIterationPlanner,
                              PolicyIterationPlanner,
                              RtdpPlanner)
from gym_mapf.solvers.rtdp import manhattan_heuristic, prioritized_value_iteration_heuristic


class GeneralPlannersTest(unittest.TestCase):
    def get_planner(self) -> Planner:
        """Return the concrete planner"""
        raise unittest.SkipTest("This is a general test case")

    def test_corridor_switch(self):
        grid = MapfGrid(['...',
                         '@.@'])
        agents_starts = ((0, 0), (0, 2))
        agents_goals = ((0, 2), (0, 0))

        # These parameters are for making sre that VI avoids collision regardless of reward efficiency
        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -0.001, 0, -1)

        planner = self.get_planner()
        policy = planner.plan(env)

        interesting_state = env.locations_to_state(((1, 1), (0, 1)))
        expected_possible_actions = [vector_action_to_integer((STAY, UP)),
                                     vector_action_to_integer((DOWN, UP))]

        self.assertIn(policy.act(interesting_state), expected_possible_actions)


class GeneralValueIterationPlannerTest(GeneralPlannersTest):
    def get_planner(self) -> Planner:
        return ValueIterationPlanner(gamma=1.0)


class GeneralPolicyIterationPlannerTest(GeneralPlannersTest):
    def get_planner(self) -> Planner:
        return PolicyIterationPlanner(gamma=1.0)


class GeneralRtdpPlannerTest(GeneralPlannersTest):
    def get_planner(self) -> Planner:
        return RtdpPlanner(prioritized_value_iteration_heuristic, 100, 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
