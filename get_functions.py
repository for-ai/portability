import json
from inspect import getmembers, isclass, isfunction, ismodule, ismethod
import tensorflow as tf
import torch
import functools


def get_functions(module, name):
    for member in getmembers(module, isfunction):
        print(name, ".", member)


module_set = set()

def recurse_module(name, framework):
    print(name)

    if framework == "torch":
        skip_list = ["torch._ops", "torch._classes", "torch.nn.quantized.modules.conv", "torch.nn.quantized.modules.functional_modules", "torch.nn.quantized.modules.utils"]
        return name not in skip_list
    elif framework == "tensorflow":
        return True

def choose_module(framework):
    if framework == "torch":
        return torch
    elif framework == "tensorflow":
        return tf

def include_module(name, framework, is_top_level):
    if framework == "torch":
        return not name.startswith("torch._")
    elif framework == "tensorflow":
        return (not is_top_level and name[0] != "_") or "tensorflow._api.v2" in name or "tensorboard.summary" in name or "tensorflow_estimator" in name or "keras" in name or "compat" in name

def iterate_module(framework, module=None, is_top_level=True, namespace_dict={}, isClass = False):
    # if module.__name__ == "torch.amp.autocast_mode":
    #     import code; code.interact(local=dict(globals(), **locals()))
    if is_top_level:
        module = choose_module(framework)
    for child in getmembers(module, isclass if isClass else ismodule):
        name = child[0]
        child = child[1]
        if child.__name__ not in module_set and ((include_module(child.__name__, framework, is_top_level) and module.__name__ in child.__name__) or isClass):
            module_set.add(child.__name__)

            if framework == "tensorflow":
                key = child.__name__.removeprefix("tensorflow._api.v2.")
            else:
                key = child.__name__
            namespace_dict[key] = {"functions": [], "classes": {}, "modules": {}, "methods": []}
            # print("child", child, hasattr(child, "__iter__"))
            # if child.__name__ =="torch.classes":
            #     import code; code.interact(local=dict(globals(), **locals()))
            # print(child)
            # if child.__name__ == "torch.distributions.lkj_cholesky":
            #     import code; code.interact(local=dict(globals(), **locals()))
            try:
                for member in getmembers(child, ismethod):
                    # print("fn", key, ".", member[0])
                    namespace_dict[key]["methods"].append(member[0])

                for member in getmembers(child, isfunction):
                    # print("fn", key, ".", member[0])
                    namespace_dict[key]["functions"].append(member[0])
                for member in getmembers(child, isclass):
                    # print("class", key, ".", member[0])
                    namespace_dict[key]["classes"][member[0]] = iterate_module(framework, child, False, {"functions": [], "classes": {}, "methods": []}, True)
                if not isClass:
                    namespace_dict[key]["modules"] = iterate_module(framework, child, False, {})
            except TypeError:
                print("EXCLUDING", child)
                pass 
        # elif is_top_level:
        #     print("EXCLUDING", child, child.__name__, child.__name__ not in module_set)
    return namespace_dict
            # if recurse_module(child.__name__, framework):

            # print("MODULE", child.__name__.removeprefix("tensorflow._api.v2."),
            #       child.__name__, name, module.__name__)

def count_torch_functions(functions_dict):
    result = {}
    for key, value in functions_dict.items():
        result[key] = {}
        result[key]["functions"] = len(value["functions"])
        result[key]["classes"] = len(value["classes"])

    f = open("torch_function_count.json", "w")
    f.write(json.dumps(result, indent=4, sort_keys=True))
    f.close()
    

def top_count_torch_functions(functions_dict):
    function_count = 0
    class_count = 0
    for key, value in functions_dict.items():
        if "modules" in value:
            module_function_count, module_class_count = top_count_torch_functions(value["modules"])
            function_count += module_function_count
            class_count += module_class_count
        class_function_count, class_class_count = top_count_torch_functions(value["classes"])
        function_count += len(value["functions"]) + class_function_count
        class_count += len(value["classes"]) + class_class_count
    return function_count, class_count

def torch_functions():
    # iterate_module("tensorflow", tf, is_top_level=True, namespace_dict=module_dict)
    framework = "torch"
    resulting_dict = iterate_module(framework, torch, is_top_level=True, namespace_dict={})
    f = open(framework + "_functions.json", "w")
    function_count, class_count = top_count_torch_functions(resulting_dict)
    print("FUNCTION COUNT", function_count, class_count)
    f.write(json.dumps(resulting_dict, indent=4, sort_keys=True))
    f.close()

def tensorflow_functions():
    # iterate_module("tensorflow", tf, is_top_level=True, namespace_dict=module_dict)
    framework = "tensorflow"
    resulting_dict = iterate_module(framework, torch, is_top_level=True, namespace_dict={})
    f = open(framework + "_functions.json", "w")
    f.write(json.dumps(resulting_dict, indent=4, sort_keys=True))
    f.close()


torch_functions()
# tensorflow_functions()

