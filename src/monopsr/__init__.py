import os


def root_dir():
    # Package root
    return os.path.dirname(os.path.realpath(__file__))


def top_dir():
    # Project top level
    return os.sep.join(root_dir().split(os.sep)[:-2])


def data_dir():
    # Data folder
    return top_dir() + '/data'


def scripts_dir():
    # Script folder
    return top_dir() + '/scripts'
