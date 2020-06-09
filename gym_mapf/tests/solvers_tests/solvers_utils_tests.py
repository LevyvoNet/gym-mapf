import unittest
import json

from gym_mapf.solvers.utils import CrossedPolicy, detect_conflict, solve_independently_and_cross, Policy
from gym_mapf.solvers.vi import ValueIterationPlanner
from gym_mapf.envs.utils import MapfGrid, get_local_view, create_mapf_env
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    DOWN, RIGHT, LEFT, STAY,
                                    ACTIONS)
from gym_mapf.solvers.rtdp import RtdpPlanner, prioritized_value_iteration_heuristic


class DictPolicy(Policy):
    def __init__(self, env, gamma, dict_policy):
        super().__init__(env, 1.0)
        self.dict_policy = dict_policy

    def act(self, s):
        return self.dict_policy[s]

    def dump_to_str(self):
        return json.dumps({'env': self.env,
                           'dict_policy': self.dict_policy})

    def load_from_str(json_str: str) -> object:
        json_obj = json.loads(json_str)
        return DictPolicy(json_obj['env'], 1.0)


def evaluate_policy(policy: Policy, n_episodes: int, max_steps: int):
    total_reward = 0
    for i in range(n_episodes):
        policy.env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            new_state, reward, done, info = policy.env.step(policy.act(policy.env.s))
            total_reward += reward
            steps += 1
            if reward == policy.env.reward_of_clash and done:
                print("clash happened, entering debug mode")
                import ipdb
                ipdb.set_trace()

    return total_reward / n_episodes


class SolversUtilsTests(unittest.TestCase):

    def test_detect_conflict_finds_classical_conflict(self):
        grid = MapfGrid(['...',
                         '@.@',
                         '...'])

        agents_starts = ((0, 0), (0, 2))
        agents_goals = ((2, 0), (2, 2))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0, 0, -1, 1, -0.01)

        policy1 = {
            0: ACTIONS.index(RIGHT),
            1: ACTIONS.index(STAY),
            2: ACTIONS.index(DOWN),
            3: ACTIONS.index(DOWN),
            4: ACTIONS.index(LEFT),
            5: ACTIONS.index(RIGHT),
            6: ACTIONS.index(LEFT),
        }

        policy2 = {
            0: ACTIONS.index(RIGHT),
            1: ACTIONS.index(RIGHT),
            2: ACTIONS.index(DOWN),
            3: ACTIONS.index(DOWN),
            4: ACTIONS.index(RIGHT),
            5: ACTIONS.index(LEFT),
            6: ACTIONS.index(STAY),
        }

        joint_policy = CrossedPolicy(env, 1.0, [DictPolicy(get_local_view(env, [0]), 1.0, policy1),
                                                DictPolicy(get_local_view(env, [1]), 1.0, policy2)])

        aux_local_env = get_local_view(env, [0])

        self.assertEqual(detect_conflict(env, joint_policy),
                         (0,
                          aux_local_env.locations_to_state(((0, 0),)),
                          1,
                          aux_local_env.locations_to_state(((0, 2),)),
                          aux_local_env.locations_to_state(((0, 1),)),
                          )
                         )

    def test_detect_conflict_return_none_when_no_conflict(self):
        grid = MapfGrid(['...',
                         '...',
                         '...'])

        agents_starts = ((0, 0), (0, 2))
        agents_goals = ((2, 0), (2, 2))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0, 0, -1, 1, -0.01)

        policy1 = {
            0: ACTIONS.index(DOWN),
            1: ACTIONS.index(DOWN),
            2: ACTIONS.index(DOWN),
            3: ACTIONS.index(DOWN),
            4: ACTIONS.index(DOWN),
            5: ACTIONS.index(DOWN),
            6: ACTIONS.index(DOWN),
            7: ACTIONS.index(DOWN),
            8: ACTIONS.index(DOWN),
        }

        policy2 = {
            0: ACTIONS.index(DOWN),
            1: ACTIONS.index(DOWN),
            2: ACTIONS.index(DOWN),
            3: ACTIONS.index(DOWN),
            4: ACTIONS.index(DOWN),
            5: ACTIONS.index(DOWN),
            6: ACTIONS.index(DOWN),
            7: ACTIONS.index(DOWN),
            8: ACTIONS.index(DOWN),
        }

        joint_policy = CrossedPolicy(env, 1.0, [DictPolicy(get_local_view(env, [0]), 1.0, policy1),
                                                DictPolicy(get_local_view(env, [1]), 1.0, policy2)])

        self.assertEqual(detect_conflict(env, joint_policy), None)

    def test_roni_scenario_with_id(self):
        # TODO: this test only pass when the first action in the ACTIONS array is STAY,
        #  fix it to work without the cheating
        grid = MapfGrid(['.@.',
                         '.@.',
                         '...'])
        agents_starts = ((0, 0), (0, 2))
        agents_goals = ((2, 0), (2, 2))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.01, -1, 1, -0.1)

        independent_joiont_policy = solve_independently_and_cross(env, [[0], [1]], ValueIterationPlanner(1.0), {})

        interesting_state = env.locations_to_state(((0, 0), (0, 2)))

        # Assert independent_joint_policy just choose the most efficient action
        self.assertEqual(independent_joiont_policy.act(interesting_state), vector_action_to_integer((DOWN, DOWN)))

        # Assert no conflict
        self.assertEqual(detect_conflict(env, independent_joiont_policy), None)

    def test_conflict_detected_for_room_scenario_with_crossed_policy(self):
        env = create_mapf_env('room-32-32-4', 1, 2, 0.1, 0.1, -1000, 0, -1)

        planner = RtdpPlanner(prioritized_value_iteration_heuristic, 100, 1.0)

        policy1 = planner.plan(get_local_view(env, [0]), {})
        policy2 = planner.plan(get_local_view(env, [1]), {})
        crossed_policy = CrossedPolicy(env, 1.0, [policy1, policy2])

        self.assertIsNot(detect_conflict(env, crossed_policy), None)

    # def test_no_conflict_detected_for_room_scenario_with_joint_policy(self):
    #     # conflict detection takes forever for this scenario for some reason
    #     env = create_mapf_env('room-32-32-4', 1, 2, 0.1, 0.1, -1000, 0, -1)
    #
    #     planner = RtdpPlanner(prioritized_value_iteration_heuristic, 100, 1.0)
    #
    #     joint_policy = planner.plan(env, {})
    #
    #     self.assertIs(detect_conflict(env, joint_policy), None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
