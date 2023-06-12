import os

def callpath(filename):
    return os.path.join(os.path.dirname(__file__), '..', filename)

def terminal_bool(arg):
    arg = arg.lower()
    return arg in ('t', 'y')