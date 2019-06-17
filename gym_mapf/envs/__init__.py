import os

MAPS_PATH = os.path.abspath(os.path.join(__file__, '../../maps/'))


def map_name_to_files(map_name):
    map_file_path = os.path.join(MAPS_PATH, map_name, '{}.map'.format(map_name))
    scen_file_path = os.path.join(MAPS_PATH, map_name, '{}-even-1.scen'.format(map_name))

    return map_file_path, scen_file_path


UP = 'UP'
RIGHT = 'RIGHT'
DOWN = 'DOWN'
LEFT = 'LEFT'
STAY = 'STAY'
POSSIBILITIES = {
    UP: (RIGHT, LEFT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (DOWN, UP),
    STAY: (STAY, STAY)
}
ACTIONS = [UP, RIGHT, DOWN, LEFT, STAY]
