from gym_mapf.utils.grid import EmptyCell, ObstacleCell

CELL_TO_CHAR = {
    EmptyCell: '.',
    ObstacleCell: ''
}


class MapfState:
    def __init__(self, map, agent_locations):
        self.map = map
        self.agent_locations = agent_locations

    def __eq__(self, other):
        return self.map == other.map and self.agent_locations == other.agent_locations

    def __hash__(self):
        return hash((hash(self.map), hash(tuple(self.agent_locations))))

    def __repr__(self):
        char_list = [
            [str(self.agent_locations.index((i, j))) if (i, j) in self.agent_locations else CELL_TO_CHAR[self.map[i, j]]
             for i in range(len(self.map))]
            for j in range(len(self.map[0]))
        ]

        lines = []
        for line in char_list:
            lines.append(''.join(line))

        return '\n'.join(lines)
