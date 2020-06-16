import unittest

from gym_mapf.solvers.rtdp import RtdpPlanner, prioritized_value_iteration_heuristic
from gym_mapf.envs.utils import create_mapf_env
from gym_mapf.solvers.utils import evaluate_policy


class RtdpPlannerTest(unittest.TestCase):
    def test_difficult_env_converges(self):
        env = create_mapf_env('room-32-32-4', 13, 2, 0, 0, -1000, 0, -1)

        # 100 iterations are not enough, 1000 are fine tough.
        planner = RtdpPlanner(prioritized_value_iteration_heuristic, 1000, 1.0)
        policy = planner.plan(env, {})

        reward, clashed = evaluate_policy(policy, 1, 1000)

        # Assert that the solution is reasonable (actually solving)
        self.assertGreater(reward, -1000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
