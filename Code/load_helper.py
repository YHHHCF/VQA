import numpy as np


def load_file(path):
    file = np.load(path, allow_pickle=True)
    file = file['arr_0']
    file = list(file)
    return file


def load_dict(path):
    dict = np.load(path, allow_pickle=True)
    dict = dict['arr_0']
    dict = dict.item()
    keys = dict.keys()
    keys = list(keys)
    return dict, keys
