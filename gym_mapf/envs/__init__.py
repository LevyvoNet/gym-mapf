import os

MAPS_PATH = os.path.abspath(os.path.join(__file__, '../../maps/'))


def map_name_to_files(map_name, scen_id):
    map_file_path = os.path.join(MAPS_PATH, map_name, '{}.map'.format(map_name))
    scen_file_path = os.path.join(MAPS_PATH, map_name, '{}-even-{}.scen'.format(map_name, scen_id))

    return map_file_path, scen_file_path
