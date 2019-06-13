class MapfState:
    def __init__(self, map, agent_locations):
        self.map = map
        self.agent_locations = agent_locations

    def __eq__(self, other):
        return self.map == other.map and self.agent_locations == other.agent_locations

    def __hash__(self):
        return hash((hash(self.map), hash(tuple(self.agent_locations))))

    def __repr__(self):
        return repr(self.agent_locations)
