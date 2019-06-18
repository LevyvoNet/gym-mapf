class ObstacleCell:
    pass


class EmptyCell:
    pass


CHAR_TO_CELL = {
    '.': EmptyCell,
    '@': ObstacleCell

}


class MapfGrid:
    def __init__(self, map_lines):
        self._map = []
        for line in map_lines:
            line = line.strip()
            new_line = [CHAR_TO_CELL[char] for char in line]
            self._map.append(new_line)

    def __getitem__(self, *args):
        if type(args[0]) == int:
            return self._map[args[0]]

        ret = self._map
        for arg in args[0]:
            ret = ret[arg]

        return ret

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)

    def __eq__(self, other):
        return self._map == other._map
