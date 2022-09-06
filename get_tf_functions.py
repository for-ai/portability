from inspect import getmembers, isfunction, ismodule
import tensorflow as tf


def get_functions(module, name):
    for member in getmembers(module, isfunction):
        print(name, ".", member)


path_set = set()


def iterate_module(module=None, isTopLevel=True):
    get_functions(module, module.__name__)
    for child in getmembers(module, ismodule):
        name = child[0]
        if "tensorflow._api.v2" in child[1].__name__ or "tensorboard.summary" in child[1].__name__ or "tensorflow_estimator" in child[1].__name__ or "keras" in child[1].__name__ or "compat" in child[1].__name__:
            path_set.add(child[1].__name__)
            get_functions(module, child[1].__name__)
            child = child[1]
            path_set.add(child.__name__)
            print("MODULE", child.__name__, name, module.__name__)
        else:
            True


iterate_module(tf)
