import json

def combine_function_tests(list, functions={}):
    for key, value in list.items():
        function = key.split(":")[0]
        if not function in functions:
            functions[function] = {"operations": [], "test_time": []}
        # print("COUNT", len(functions[function]["operations"]))
        functions[function]["operations"] += value["operations"]
        functions[function]["test_time"].append(value["test_time"])
    return functions

def open_file(framework, device, n, function_list={}):
    if framework == "tensorflow":
        f = open(f'./tensorflow_timings/{device}_vm/{device}/{device}_{n}.json')
    elif framework == "torch":
        f = open(f'./pytorch_timings/{device}_vm/{device}_{n}.json')
    function_list = combine_function_tests(json.load(f), function_list)
    f.close()
    return function_list

    # print("FUNCTION LIST", len(tpu_function_list["CheckpointSaverHook_test.py"]["operations"]))
    # f = open("./tensorflow_timings/tpu_vm/tpu/tpu_2.json")
    # function_list = combine_function_tests(json.load(f), tpu_function_list)
    # f.close()
    # print("FUNCTION LIST", len(tpu_function_list["CheckpointSaverHook_test.py"]["operations"]))
    # f = open("./tensorflow_timings/tpu_vm/tpu/tpu_3.json")
    # function_list = combine_function_tests(json.load(f), tpu_function_list)
    # f.close()
    # print("FUNCTION LIST", len(tpu_function_list["CheckpointSaverHook_test.py"]["operations"]))


def fetch_data(framework):
    tpu_function_list = open_file(framework, "tpu", 1, {})
    tpu_function_list = open_file(framework, "tpu", 2, tpu_function_list)
    tpu_function_list = open_file(framework, "tpu", 3, tpu_function_list)
    gpu_function_list = open_file(framework, "gpu", 1, {})
    gpu_function_list = open_file(framework, "gpu", 2, gpu_function_list)
    gpu_function_list = open_file(framework, "gpu", 3, gpu_function_list)
    function_keys = gpu_function_list.keys()
    return gpu_function_list, tpu_function_list, function_keys

