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

def fetch_data(framework):
    framework = "tensorflow"
    frameworkTitle = framework.capitalize()
    print("TPU")
    f = open("./tensorflow_timings/tpu_vm/tpu/tpu_1.json")
    tpu_function_list = combine_function_tests(json.load(f))
    f.close()
    print("FUNCTION LIST", len(tpu_function_list["CheckpointSaverHook_test.py"]["operations"]))
    f = open("./tensorflow_timings/tpu_vm/tpu/tpu_2.json")
    tpu_function_list = combine_function_tests(json.load(f), tpu_function_list)
    f.close()
    print("FUNCTION LIST", len(tpu_function_list["CheckpointSaverHook_test.py"]["operations"]))
    f = open("./tensorflow_timings/tpu_vm/tpu/tpu_3.json")
    tpu_function_list = combine_function_tests(json.load(f), tpu_function_list)
    f.close()
    print("FUNCTION LIST", len(tpu_function_list["CheckpointSaverHook_test.py"]["operations"]))

    print("GPU")
    f = open("./tensorflow_timings/gpu_vm/gpu/gpu_1.json")
    gpu_function_list = combine_function_tests(json.load(f), {})
    f.close()
    print("FUNCTION LIST", len(gpu_function_list["CheckpointSaverHook_test.py"]["operations"]))
    f = open("./tensorflow_timings/gpu_vm/gpu/gpu_2.json")
    gpu_function_list = combine_function_tests(json.load(f), gpu_function_list)
    f.close()
    print("FUNCTION LIST", len(gpu_function_list["CheckpointSaverHook_test.py"]["operations"]))
    f = open("./tensorflow_timings/gpu_vm/gpu/gpu_3.json")
    gpu_function_list = combine_function_tests(json.load(f), gpu_function_list)
    f.close()
    print("FUNCTION LIST", len(gpu_function_list["CheckpointSaverHook_test.py"]["operations"]))
    function_keys = gpu_function_list.keys()
    return gpu_function_list, tpu_function_list, function_keys

