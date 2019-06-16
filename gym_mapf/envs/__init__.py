import os

EMPTY_8_8_MAP_FILE = os.path.abspath(os.path.join(__file__, '../../../maps/empty-8-8/empty-8-8.map'))
EMPTY_8_8_SCEN_FILE = os.path.abspath(os.path.join(__file__, '../../../maps/empty-8-8/empty-8-8-even-1.scen'))
BERLIN_1_256_MAP_FILE = os.path.abspath(os.path.join(__file__, '../../../maps/Berlin_1_256/Berlin_1_256.map'))
BERLIN_1_256_SCEN_FILE = os.path.abspath(os.path.join(__file__, '../../../maps/Berlin_1_256/Berlin_1_256-even-1.scen'))
MAP_NAME_TO_FILES = {
    'empty-8-8': (EMPTY_8_8_MAP_FILE, EMPTY_8_8_SCEN_FILE),
    'berlin-1-256': (BERLIN_1_256_MAP_FILE, BERLIN_1_256_SCEN_FILE)

}

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
