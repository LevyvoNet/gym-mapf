"""A module for general policy class."""
from gym import Env
from abc import abstractmethod, ABCMeta


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
    def dump_to_json(self):
        """Dump policy parameters to a JSON in a reproducible way

        Returns:
            str. JSON representation of the policy.
        """

    @staticmethod
    @abstractmethod
    def load_from_json(cls, json_str):
        """Load policy from JSON

        Args:
            cls (type): the type of policy to load, need to be known in advance.
            json_str (str): a string with a JSON representation dumped by the policy dump_to_json method.

        Returns:
            Policy. The policy represented by the given JSON.
        """
