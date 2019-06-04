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
        with open(map_file, 'r') as f:
            lines = iter(f)
            for _ in range(4):  # skip first 4 lines
                next(lines)
            for line in lines:
                line = line.strip()
                new_line = [CHAR_TO_CELL[char] for char in line]
                self._map.append(new_line)

    def __getitem__(self, item):
        row, col = item
        return self._map[row][col]

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)
