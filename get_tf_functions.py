import json
from inspect import getmembers, isclass, isfunction, ismodule
import tensorflow as tf


def get_functions(module, name):
    for member in getmembers(module, isfunction):
        print(name, ".", member)


module_dict = {}
module_set = set()


def iterate_module(module=None, isTopLevel=True, dict={}, iterating_type=ismodule):
    for child in getmembers(module, iterating_type):
        name = child[0]
        child = child[1]
        print(name, child)
        if child.__name__ not in module_set and ((not isTopLevel and name[0] != "_") or "tensorflow._api.v2" in child.__name__ or "tensorboard.summary" in child.__name__ or "tensorflow_estimator" in child.__name__ or "keras" in child.__name__ or "compat" in child.__name__):
            module_set.add(child.__name__)

            key = child.__name__.removeprefix("tensorflow._api.v2.")
            dict[key] = {"functions": [], "classes": {}, "modules": {}}
            for member in getmembers(child, isfunction):
                print("fn", key, ".", member[0])
                dict[key]["functions"].append(member[0])
            for member in getmembers(child, isclass):
                print("class", key, ".", member[0])
                dict[key]["classes"][member[0]] = {"functions": [], "classes": {}}
                iterate_module(child, False, dict[key]["modules"][member[0]], isclass)
            iterate_module(child, False, dict[key]["modules"])

            # print("MODULE", child.__name__.removeprefix("tensorflow._api.v2."),
            #       child.__name__, name, module.__name__)

    f = open("tf_functions.json", "w")
    f.write(json.dumps(module_dict, indent=4, sort_keys=True))
    f.close()


iterate_module(tf, isTopLevel=True, dict=module_dict)
