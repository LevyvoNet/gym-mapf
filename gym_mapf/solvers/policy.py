"""A module for general policy class."""
import json

import numpy as np
from gym import Env
from abc import abstractmethod, ABCMeta

from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.envs.utils import mapf_env_load_from_json


class Policy(metaclass=ABCMeta):
    def __init__(self, env: Env):
        self.env = env

    @abstractmethod
    def act(self, s):
        """Return the policy action for a given state

        Args:
            s (int): a state of self environment

        Returns:
            int. The best action according to that policy.
        """

    @abstractmethod
    def dump_to_str(self):
        """Dump policy parameters to a JSON in a reproducible way

        Returns:
            str. JSON representation of the policy.
        """

    @staticmethod
    @abstractmethod
    def load_from_str(json_str: str) -> object:
        """Load policy from JSON

        Args:
            json_str (str): a string with a JSON representation dumped by the policy dump_to_json method.

        Returns:
            Policy. The policy represented by the given JSON.
        """


class TabularValueFunctionPolicy(Policy):
    def __init__(self, env: MapfEnv, gamma):
        super().__init__(env)
        self.v = np.zeros(env.nS)
        self.tabular_policy = np.zeros(env.nS)
        self.gamma = gamma

    def act(self, s):
        return int(self.tabular_policy[s])

    def dump_to_str(self):
        return json.dumps({'v': self.v.tolist(),
                           'gamma': self.gamma,
                           'env': self.env})

    @staticmethod
    def load_from_str(json_str: str) -> Policy:
        json_obj = json.loads(json_str)
        env = mapf_env_load_from_json(json_obj['env'])
        v = np.asarray(json_obj['v'])
        gamma = json_obj['gamma']
        vi_policy = TabularValueFunctionPolicy(env, gamma)
        vi_policy.v = v
        # TODO: implement extraction of policy from value function, might be complicated
        vi_policy.tabular_policy = None

        return vi_policy
