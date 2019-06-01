import numpy as np


def load_file(path):
    file = np.load(path)
    file = file['arr_0']
    file = list(file)
    return file


def load_dict(path):
    dict = np.load(path)
    dict = dict['arr_0']
    dict = dict.item()
    keys = dict.keys()
    keys = list(keys)
    return dict, keys
