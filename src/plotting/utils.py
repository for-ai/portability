import json
import numpy as np


def combine_function_tests(
    list, functions={}, error_list={}, by_function=True, consolidation_method="concat"
):
    for error_value in error_list["tests"]:
        key = error_value["nodeid"]
        with_suffix = (
            error_value["nodeid"]
            .removeprefix("src/tensorflow_tests_reduced/")
            .removeprefix("src/pytorch_tests_reduced/")
            .removeprefix("src/jax_tests_reduced/")
            .split(":")[0]
        )
        # if framework == "jax":
        #     import code

        #     code.interact(local=dict(globals(), **locals()))

        test = key.split(":")[-1]
        key = f"{with_suffix}:{test}"
        print("KEY", key)
        if key in list:
            if error_value["outcome"] == "passed" and "_cpu" not in key:
                value = list[key]

                if not by_function:
                    key = key.split(":")[0]
                # for key, value in list.items():
                # if error_value["outcome"] == "passed":
                if not key in functions:
                    functions[key] = {"operations": [], "test_time": []}
                # print("COUNT", len(functions[function]["operations"]))
                if consolidation_method == "concat":
                    functions[key]["operations"] += value["operations"]
                elif consolidation_method == "mean":
                    if len(functions[key]["operations"]) > 0:
                        for i, op in enumerate(value["operations"]):
                            functions[key]["operations"][i] = np.append(
                                functions[key]["operations"][i], op
                            )
                    else:
                        for i, op in enumerate(value["operations"]):
                            functions[key]["operations"].append(
                                np.array(value["operations"][i], dtype=np.float64)
                            )

                functions[key]["test_time"].append(value["test_time"])
    return functions


def open_file(framework, device, n, function_list={}, consolidation_method="concat"):
    if framework == "tensorflow" or framework == "jax":
        framework_dir = f"{framework}_timings"
        f = open(f"./{framework}_timings/{device}_vm/{device}/{device}_{n}.json")
    elif framework == "torch":
        framework_dir = "pytorch_timings"
        f = open(f"./pytorch_timings/{device}_vm/{device}_{n}.json")

    time_list = json.load(f)
    error_list_handler = open(f"./{framework_dir}/{device}_vm/test_failure_report.json")
    error_list = json.load(error_list_handler)

    function_list = combine_function_tests(
        time_list, function_list, error_list, consolidation_method=consolidation_method
    )
    new_function_list = {}
    for key in function_list.keys():
        if function_list[key]["operations"] != []:
            new_function_list[key] = function_list[key]
    function_list = new_function_list
    f.close()
    error_list_handler.close()
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


def fetch_data(framework, by_function=True, consolidation_method="concat"):
    tpu_function_list = open_file(framework, "tpu", 1, {}, consolidation_method)
    tpu_function_list = open_file(
        framework, "tpu", 2, tpu_function_list, consolidation_method
    )
    tpu_function_list = open_file(
        framework, "tpu", 3, tpu_function_list, consolidation_method
    )

    gpu_function_list = open_file(framework, "gpu", 1, {}, consolidation_method)
    gpu_function_list = open_file(
        framework, "gpu", 2, gpu_function_list, consolidation_method
    )
    gpu_function_list = open_file(
        framework, "gpu", 3, gpu_function_list, consolidation_method
    )

    new_tpu_function_list = {}
    new_gpu_function_list = {}

    if framework == "torch":
        for key in tpu_function_list.keys():
            new_tpu_function_list[key.split("_xla")[0]] = tpu_function_list[key]
        for key in gpu_function_list.keys():
            new_gpu_function_list[key.split("_cuda")[0]] = gpu_function_list[key]

        gpu_function_list = new_gpu_function_list
        tpu_function_list = new_tpu_function_list
        new_gpu_function_list = {}
        new_tpu_function_list = {}

    for key in tpu_function_list.keys():
        if key in gpu_function_list.keys():
            new_tpu_function_list[key] = tpu_function_list[key]
            for i, op in enumerate(new_tpu_function_list[key]["operations"]):
                np.mean(new_tpu_function_list[key]["operations"][i])

    for key in gpu_function_list.keys():
        if key in tpu_function_list.keys():
            new_gpu_function_list[key] = gpu_function_list[key]
            for i, op in enumerate(new_gpu_function_list[key]["operations"]):
                np.mean(new_gpu_function_list[key]["operations"][i])

    if by_function:
        gpu_function_list = new_gpu_function_list
        tpu_function_list = new_tpu_function_list
        new_gpu_function_list = {}
        new_tpu_function_list = {}
        for key in tpu_function_list.keys():
            func_key = key.split(":")[0].removesuffix(".py")
            if not func_key in new_tpu_function_list:
                new_tpu_function_list[func_key] = {"operations": [], "test_time": []}
            new_tpu_function_list[func_key]["operations"] += tpu_function_list[key][
                "operations"
            ]

        for key in gpu_function_list.keys():
            func_key = key.split(":")[0].removesuffix(".py")
            if not func_key in new_gpu_function_list:
                new_gpu_function_list[func_key] = {"operations": [], "test_time": []}
            new_gpu_function_list[func_key]["operations"] += gpu_function_list[key][
                "operations"
            ]

        # if not by_function:
    function_keys = new_gpu_function_list.keys()

    return new_gpu_function_list, new_tpu_function_list, function_keys
