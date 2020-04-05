import os

MAPS_PATH = os.path.abspath(os.path.join(__file__, '../../maps/'))


def map_name_to_files(map_name, scen_id):
    map_file_path = os.path.join(MAPS_PATH, map_name, '{}.map'.format(map_name))
    scen_file_path = os.path.join(MAPS_PATH, map_name, '{}-even-{}.scen'.format(map_name, scen_id))

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
ACTIONS = [STAY, UP, RIGHT, DOWN, LEFT]
ACTIONS_TO_INT = {
    a: ACTIONS.index(a) for a in ACTIONS
}


def integer_to_vector(x, options_per_element, n_elements, index_to_element):
    """Return a vector representing an action/state from a given integer.

    Args:
        x (int): the integer to convert.
        n_options_per_element(int): number of options for each element in the vector.
        n_elements (int): the number of elements in the vector to return.
        index_to_element(int=>any): function which converts an integer represents a single option in one of the
            vector elements and return anything that vector contains. For example, a function which returns 'UP' for 0,
            1 for 'RIGHT',etc. Or a function which returns (2,2) given 10 for a 4x4 grid ((2,2) is the 10-th cell of that grid).
    """
    return integer_to_vector_multiple_numbers(x, options_per_element, n_elements, index_to_element)


def vector_to_integer(v, options_per_element, element_to_index):
    return vector_to_integer_multiple_numbers(v, options_per_element, element_to_index)


def integer_to_vector_multiple_numbers(x, n_options_per_element, n_elements, index_to_element):
    """Return a vector representing an action/state from a given integer.

    Args:
        x (int): the integer to convert.
        n_options_per_element(list): number of options for each element in the vector.
        n_elements (int): the number of elements in the vector to return.
        index_to_element(int=>any): function which converts an integer represents a single option in one of the
            vector elements and return anything that vector contains. For example, a function which returns 'UP' for 0,
            1 for 'RIGHT',etc. Or a function which returns (2,2) given 10 for a 4x4 grid ((2,2) is the 10-th cell of that grid).
    """
    ret = tuple()
    for i in range(0, n_elements):
        option_index = x % n_options_per_element[i]
        ret = ret + (index_to_element(option_index),)
        x //= n_options_per_element[i]

    return ret


def vector_to_integer_multiple_numbers(v, n_options_per_element, element_to_index):
    sum = 0
    mul = 1
    for i in range(len(v)):
        if i != 0:
            mul *= n_options_per_element[i - 1]

        sum += element_to_index(v[i]) * mul

    return sum
