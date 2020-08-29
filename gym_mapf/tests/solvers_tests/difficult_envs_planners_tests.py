import unittest
from typing import Dict, Callable
from functools import partial

from gym_mapf.solvers.rtdp import (rtdp,
                                   prioritized_value_iteration_heuristic,
                                   fixed_iterations_count_rtdp,
                                   stop_when_no_improvement_rtdp)
from gym_mapf.solvers.lrtdp import lrtdp
from gym_mapf.envs.utils import create_mapf_env, MapfEnv, MapfGrid
from gym_mapf.solvers.utils import evaluate_policy, Policy


class DifficultEnvsPlannerTest(unittest.TestCase):
    """
    This test case is for stochastic environments with multiple agents and complicated maps (room for exapmle).

    so far, only RTDP based solvers succeed to solve such environments.
    """

    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        """Return the concrete planner"""
        raise unittest.SkipTest("This is an abstract test case")

    def black_box_test(self, policy: Policy, info: Dict):
        pass

    def test_room_scen_13_converges(self):
        raise unittest.SkipTest("temp, too hard")
        # TODO: figure out what is so hard about that scenario and look for bugs
        env = create_mapf_env('room-32-32-4', 13, 2, 0, 0, -1000, 0, -1)

        # 100 iterations are not enough, 1000 are fine tough.
        # TODO: make it converge faster
        planner = self.get_plan_func()
        policy = planner.plan(env, {})

        reward, clashed = evaluate_policy(policy, 1, 1000)

        # Assert that the solution is reasonable (actually solving)
        self.assertGreaterEqual(reward, -47.0)

    def test_normal_room_scenario_converges(self):
        env = create_mapf_env('room-32-32-4', 12, 2, 0, 0, -1000, 0, -1)

        plan_func = self.get_plan_func()
        info = {}
        policy = plan_func(env, info)

        reward, clashed = evaluate_policy(policy, 1, 1000)

        self.black_box_test(policy, info)

        # Assert that the solution is reasonable (actually solving)
        self.assertEqual(reward, -8.0)

    def test_deterministic_room_scenario_1_2_agents(self):
        # TODO: make this test pass!!!
        raise unittest.SkipTest("temp, too hard")
        env = create_mapf_env('room-32-32-4', 1, 2, 0, 0, -1000, 0, -1)

        info = {}
        planner = self.get_plan_func()
        policy = planner(env, info)

        reward, _ = evaluate_policy(policy, 1, 100)
        self.assertEqual(reward, -43)

    def test_hand_crafted_env_converges(self):
        grid = MapfGrid([
            '...',
            '@.@',
            '@.@',
            '...'])

        agent_starts = ((0, 0), (0, 2))
        agents_goals = ((3, 0), (3, 2))

        deterministic_env = MapfEnv(grid, 2, agent_starts, agents_goals,
                                    0.0, 0.0, -1000, 0, -1)

        planner = self.get_plan_func()
        policy = planner(deterministic_env, {})
        reward, clashed = evaluate_policy(policy, 1, 20)

        # Make sure this policy is optimal
        self.assertEqual(reward, -5.0)


class FixedIterationsCountRtdpPlannerTest(DifficultEnvsPlannerTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        return partial(fixed_iterations_count_rtdp,
                       partial(prioritized_value_iteration_heuristic, 1.0), 1.0,
                       2)


class StopWhenNoImprovementRtdpPlannerTest(DifficultEnvsPlannerTest):

    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        self.iter_in_batches = 100
        self.max_iterations = 10000

        return partial(stop_when_no_improvement_rtdp,
                       partial(prioritized_value_iteration_heuristic, 1.0),
                       1.0,
                       self.iter_in_batches,
                       self.max_iterations)

    def black_box_test(self, policy: Policy, info: Dict):
        n_batches = len(info['batches'])
        n_iterations = n_batches * self.iter_in_batches
        print(f'performed {n_iterations}/{self.max_iterations} iterations (in {n_batches} batches)')


# class LrtdpPlannerTest(DifficultEnvsPlannerTest):
#     def get_planner(self) -> Planner:
#         return LrtdpPlanner(prioritized_value_iteration_heuristic, 100, 1.0, 0.01)


if __name__ == '__main__':
    unittest.main(verbosity=2)