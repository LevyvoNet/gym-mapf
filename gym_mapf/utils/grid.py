class ObstacleCell:
    pass


class EmptyCell:
    pass


CHAR_TO_CELL = {
    '.': EmptyCell,
    '@': ObstacleCell

}


class MapfGrid:
    def __init__(self, map_file):
        self._map = []
        for line in open(map_file, 'r'):
            new_line = [CHAR_TO_CELL[char]() for char in line]
            self._map.append(new_line)

    def __getitem__(self, item):
        row, col = item
        return self._map[row][col]

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)
