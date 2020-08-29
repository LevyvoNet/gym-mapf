import unittest
from functools import partial
from typing import Dict, Callable

from gym_mapf.envs.utils import MapfGrid, create_mapf_env
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    UP, DOWN, STAY)
from gym_mapf.solvers.utils import Policy, evaluate_policy
from gym_mapf.solvers import (value_iteration,
                              policy_iteration,
                              rtdp,
                              id,
                              lrtdp,
                              fixed_iterations_count_rtdp,
                              stop_when_no_improvement_rtdp)
from gym_mapf.solvers.rtdp import manhattan_heuristic, prioritized_value_iteration_heuristic


class EasyEnvironmentsPlannersTest(unittest.TestCase):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        """Return the concrete planner"""
        raise unittest.SkipTest("This is an abstract test case")

    def test_corridor_switch(self):
        grid = MapfGrid(['...',
                         '@.@'])
        agents_starts = ((0, 0), (0, 2))
        agents_goals = ((0, 2), (0, 0))

        # These parameters are for making sre that VI avoids collision regardless of reward efficiency
        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -0.001, -1, -1)

        info = {}
        plan_func = self.get_plan_func()
        policy = plan_func(env, info)

        interesting_state = env.locations_to_state(((1, 1), (0, 1)))
        expected_possible_actions = [vector_action_to_integer((STAY, UP)),
                                     vector_action_to_integer((DOWN, UP))]

        self.assertIn(policy.act(interesting_state), expected_possible_actions)


class EasyEnvironmentsValueIterationPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        return partial(value_iteration, 1.0)


class EasyEnvironmentsPolicyIterationPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        return partial(policy_iteration, 1.0)


class EasyEnvironmentsFixedIterationsCountRtdpPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        return partial(fixed_iterations_count_rtdp,
                       partial(prioritized_value_iteration_heuristic, 1.0),
                       1.0,
                       100)


class EasyEnvironmentsStopWhenNoImprovementRtdpPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        return partial(stop_when_no_improvement_rtdp,
                       partial(prioritized_value_iteration_heuristic, 1.0),
                       1.0,
                       10,
                       20)


class GeneralIdPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        low_level_planner = partial(value_iteration, 1.0)
        return partial(id, low_level_planner)


# class GeneralLrdtpPlannerTest(EasyEnvironmentsPlannersTest):
#     def get_planner(self) -> Planner:
#         return LrtdpPlanner(prioritized_value_iteration_heuristic, 1000, 1.0, 0.00001)


if __name__ == '__main__':
    unittest.main(verbosity=2)
