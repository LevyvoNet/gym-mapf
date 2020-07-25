import unittest

from gym_mapf.solvers.rtdp import RtdpPlanner, prioritized_value_iteration_heuristic
from gym_mapf.solvers.lrtdp import LrtdpPlanner
from gym_mapf.envs.utils import create_mapf_env, MapfEnv, MapfGrid
from gym_mapf.solvers.utils import evaluate_policy
from gym_mapf.solvers.utils import Planner


class DifficultEnvsPlannerTest(unittest.TestCase):
    """
    This test case is for stochastic environments with multiple agents and complicated maps (room for exapmle).

    so far, only RTDP based solvers succeed to solve such environments.
    """

    def get_planner(self) -> Planner:
        """Return the concrete planner"""
        raise unittest.SkipTest("This is an abstract test case")

    def test_room_scen_13_converges(self):
        raise unittest.SkipTest("temp, too hard")
        # TODO: figure out what is so hard about that scenario and look for bugs
        env = create_mapf_env('room-32-32-4', 13, 2, 0, 0, -1000, 0, -1)

        # 100 iterations are not enough, 1000 are fine tough.
        # TODO: make it converge faster
        planner = self.get_planner()
        policy = planner.plan(env, {})

        reward, clashed = evaluate_policy(policy, 1, 1000)

        # Assert that the solution is reasonable (actually solving)
        self.assertGreaterEqual(reward, -47.0)

    def test_normal_room_scenario_converges(self):
        env = create_mapf_env('room-32-32-4', 12, 2, 0, 0, -1000, 0, -1)

        # 100 iterations are not enough, 1000 are fine tough.
        # TODO: make it converge faster
        planner = self.get_planner()
        policy = planner.plan(env, {})

        reward, clashed = evaluate_policy(policy, 1, 1000)

        # Assert that the solution is reasonable (actually solving)
        self.assertEqual(reward, -8.0)

    def test_hand_crafted_env_converges(self):
        grid = MapfGrid([
            '...',
            '@.@',
            '@.@',
            '...'])

        agent_starts = ((0, 0), (0, 2))
        agents_goals = ((3, 0), (3, 2))

        determinstic_env = MapfEnv(grid, 2, agent_starts, agents_goals,
                                   0.0, 0.0, -1000, 0, -1)

        planner = self.get_planner()
        policy = planner.plan(determinstic_env, {})
        reward, clashed = evaluate_policy(policy, 1, 20)

        # Make sure this policy is optimal
        self.assertEqual(reward, -5.0)


class RtdpPlannerTest(DifficultEnvsPlannerTest):
    def get_planner(self) -> Planner:
        return RtdpPlanner(prioritized_value_iteration_heuristic, 2, 1.0)


class LrtdpPlannerTest(DifficultEnvsPlannerTest):
    def get_planner(self) -> Planner:
        return LrtdpPlanner(prioritized_value_iteration_heuristic, 1000, 1.0, 0.0001)


if __name__ == '__main__':
    unittest.main(verbosity=2)
