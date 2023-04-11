import numpy as np
import csv
from os.path import join, isdir


def parse_gta(load_dir, seq_idx):

    print(seq_idx)
    """Parse all GTA raw data"""
    load_seq_dir = join(load_dir, 'seq_{:08d}'.format(seq_idx))

    csv_load_pathname = join(load_seq_dir, 'peds.csv')
    parsed_csv_dict = parse_csv(csv_load_pathname)

    log_load_pathname = join(load_seq_dir, 'log.txt')
    parsed_log_dict = parse_log(log_load_pathname)

    scenario_file_pathname = join(load_seq_dir, 'vid_{:08d}.txt'.format(seq_idx))
    parsed_scenario_dict = parse_scenario_file(scenario_file_pathname)

    print('parse_gta good')

    parsed_dict = {
        **parsed_csv_dict,
        **parsed_log_dict,
        **parsed_scenario_dict
    }

    return parsed_dict