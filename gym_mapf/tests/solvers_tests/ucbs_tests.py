import unittest

from gym_mapf.solvers.ucbs import best_joint_policy_under_constraint
from gym_mapf.solvers.utils import cross_policies, detect_conflict
from gym_mapf.envs.utils import MapfGrid
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    vector_state_to_integer,
                                    integer_state_to_vector,
                                    integer_action_to_vector,
                                    UP, DOWN, RIGHT, LEFT, STAY,
                                    ACTIONS)


class UcbsTests(unittest.TestCase):
    def test_detect_conflict_finds_classical_conflict(self):
        grid = MapfGrid(['...',
                         '@.@',
                         '...'])

        agents_starts = vector_state_to_integer(grid, ((0, 0), (0, 2)))
        agents_goals = vector_state_to_integer(grid, ((2, 0), (2, 2)))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0, 0, -1, 1, -0.01)

        policy1 = {
            0: ACTIONS.index(RIGHT),
            1: ACTIONS.index(DOWN),
            2: ACTIONS.index(LEFT),
            3: ACTIONS.index(RIGHT),
            4: ACTIONS.index(DOWN),
            5: ACTIONS.index(LEFT),
            6: ACTIONS.index(STAY),
            7: ACTIONS.index(LEFT),
            8: ACTIONS.index(LEFT),
        }

        policy2 = {
            0: ACTIONS.index(RIGHT),
            1: ACTIONS.index(DOWN),
            2: ACTIONS.index(LEFT),
            3: ACTIONS.index(RIGHT),
            4: ACTIONS.index(DOWN),
            5: ACTIONS.index(LEFT),
            6: ACTIONS.index(RIGHT),
            7: ACTIONS.index(RIGHT),
            8: ACTIONS.index(STAY),
        }

        n_local_states_per_agent = len(grid[0]) * len(grid)
        joint_policy = cross_policies([policy1.get, policy2.get], [n_local_states_per_agent, n_local_states_per_agent])

        self.assertEqual(detect_conflict(env, joint_policy),
                         (0,
                          vector_state_to_integer(env.grid, ((0, 0),)),
                          1,
                          vector_state_to_integer(env.grid, ((0, 2),)),
                          vector_state_to_integer(env.grid, ((0, 1),)),
                          )
                         )

    def test_detect_conflict_return_none_when_no_conflict(self):
        grid = MapfGrid(['...',
                         '...',
                         '...'])

        agents_starts = vector_state_to_integer(grid, ((0, 0), (0, 2)))
        agents_goals = vector_state_to_integer(grid, ((2, 0), (2, 2)))

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

        n_local_states_per_agent = len(grid[0]) * len(grid)
        joint_policy = cross_policies([policy1.get, policy1.get], [n_local_states_per_agent, n_local_states_per_agent])

        self.assertEqual(detect_conflict(env, joint_policy),
                         None)

    def test_find_best_policies_with_no_constraints(self):
        grid = MapfGrid(['...',
                         '@.@',
                         '...'])

        agents_starts = vector_state_to_integer(grid, ((0, 0), (0, 2)))
        agents_goals = vector_state_to_integer(grid, ((2, 0), (2, 2)))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0, 0, -1, 1, -0.01)

        sum_exptected_reward, joint_policy = best_joint_policy_under_constraint(env, [[], []])

        self.assertEqual(joint_policy(agents_starts), vector_action_to_integer((RIGHT, LEFT)))

    def test_find_best_policies_with_one_constraint(self):
        """Test that adding a constraint changes the selected policies."""
        grid = MapfGrid(['...',
                         '@.@',
                         '...'])

        agents_starts = vector_state_to_integer(grid, ((0, 0), (0, 2)))
        agents_goals = vector_state_to_integer(grid, ((2, 0), (2, 2)))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -1, 1, -0.01)

        constraint = (0,
                      vector_state_to_integer(env.grid, ((0, 0),)),
                      1,
                      vector_state_to_integer(env.grid, ((0, 2),)),
                      vector_state_to_integer(env.grid, ((0, 1),)),
                      )
        sum_exptected_reward, joint_policy = best_joint_policy_under_constraint(env, [[constraint],
                                                                                      []])

        best_action = joint_policy(agents_starts)

        # agent 0 must take an action which has no chance to get him to position (0,1).
        # that means he must not move to the right, and also up or down (because up and down might
        # turn out to be right).
        self.assertIn(best_action, [vector_action_to_integer((STAY, LEFT)),
                                    vector_action_to_integer((LEFT, LEFT))])
