import unittest
import json

from gym_mapf.solvers.policy import Policy
from gym_mapf.solvers.utils import CrossedPolicy, detect_conflict, solve_independently_and_cross
from gym_mapf.solvers.vi import value_iteration_planning
from gym_mapf.envs.utils import MapfGrid, get_local_view
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    DOWN, RIGHT, LEFT, STAY,
                                    ACTIONS)


class DictPolicy(Policy):
    def __init__(self, env, dict_policy):
        super().__init__(env)
        self.dict_policy = dict_policy

    def act(self, s):
        return self.dict_policy[s]

    def dump_to_str(self):
        return json.dumps({'env': self.env,
                           'dict_policy': self.dict_policy})

    def load_from_str(json_str: str) -> object:
        json_obj = json.loads(json_str)
        return DictPolicy(json_obj['env'], json_obj['dict_policy'])


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

        joint_policy = CrossedPolicy(env, [DictPolicy(get_local_view(env, [0]), policy1),
                                           DictPolicy(get_local_view(env, [1]), policy2)])

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

        joint_policy = CrossedPolicy(env, [DictPolicy(get_local_view(env, [0]), policy1),
                                          DictPolicy(get_local_view(env, [1]), policy2)])

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

        independent_joiont_policy = solve_independently_and_cross(env, [[0], [1]], value_iteration_planning)

        interesting_state = env.locations_to_state(((0, 0), (0, 2)))

        # Assert independent_joint_policy just choose the most efficient action
        self.assertEqual(independent_joiont_policy.act(interesting_state), vector_action_to_integer((DOWN, DOWN)))

        # Assert no conflict
        self.assertEqual(detect_conflict(env, independent_joiont_policy), None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
