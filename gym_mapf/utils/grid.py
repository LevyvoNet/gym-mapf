class ObstacleCell:
    pass


class EmptyCell:
    pass


CHAR_TO_CELL = {
    '.': EmptyCell,
    '@': ObstacleCell

}


class MapfGrid:
    # TODO: inject the map_file data in another way, make this c'tor independent on the file system
    def __init__(self, map_file):
        self._map = []
        with open(map_file, 'r') as f:
            lines = iter(f)
            for _ in range(4):  # skip first 4 lines
                next(lines)
            for line in lines:
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
