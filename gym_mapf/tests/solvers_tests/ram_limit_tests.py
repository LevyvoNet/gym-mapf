import unittest
from functools import partial
from typing import Dict, Callable
from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.envs.utils import create_mapf_env
from gym_mapf.solvers.utils import Policy
from gym_mapf.solvers import (value_iteration,
                              prioritized_value_iteration,
                              policy_iteration)


class RamLimitTest(unittest.TestCase):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        """Return the concrete planner"""
        raise unittest.SkipTest("This is an abstract test case")

    def test_multiple_agents_env(self):
        """Assert that when trying to solver a large environment we are not exceeding the RAM limit."""
        # Note the large number of agents
        env = create_mapf_env('room-32-32-4', 12, 6, 0.1, 0.1, -1000, -1, -1)

        info = {}
        plan_func = self.get_plan_func()
        policy = plan_func(env, info)

        self.assertIs(policy, None)
        self.assertEqual(info['end_reason'], 'out_of_memory')


class ValueIterationRamLimitTest(RamLimitTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        return partial(value_iteration, 1.0)


class PrioritizedValueIterationRamLimitTest(RamLimitTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        return partial(prioritized_value_iteration, 1.0)


class PolicyIterationRamLimitTest(RamLimitTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        return partial(policy_iteration, 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
