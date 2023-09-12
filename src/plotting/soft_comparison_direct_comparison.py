import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import numpy as np


# for key in function_keys:
#      if len(gpu_function_list[key]["operations"]) != len(cpu_vm_tpu_function_list[key]["operations"]) or len(tpu_function_list[key]["operations"]) != len(cpu_vm_tpu_function_list[key]["operations"]) or len(gpu_function_list[key]["operations"]) != len(tpu_function_list[key]["operations"]):
#         print("NON MATCHING KEYS", key, len(cpu_vm_tpu_function_list[key]["operations"]), len(gpu_function_list[key]["operations"]), len(tpu_function_list[key]["operations"]))
# function_keys = set(function_keys.map(lambda x: x.split(":")[0]))


def combine_function_tests(list, functions={}):
    for key, value in list.items():
        function = key.split(":")[0]
        if not function in functions:
            functions[function] = {"operations": [], "test_time": []}
        # print("COUNT", len(functions[function]["operations"]))
        functions[function]["operations"] += value["operations"]
        functions[function]["test_time"].append(value["test_time"])
    return functions


def open_file(framework, device, n, function_list={}, soft=False):
    if framework == "tensorflow" or framework == "jax":
        if soft:
            f = open(f"./{framework}_timings/soft_placement/{device}/{device}_{n}.json")
        else:
            f = open(f"./{framework}_timings/{device}_vm/{device}/{device}_{n}.json")
    elif framework == "torch":
        f = open(f"./pytorch_timings/{device}_vm/{device}_{n}.json")
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


def fetch_data(framework, soft=False):
    tpu_function_list = open_file(framework, "tpu", 1, {}, soft)
    tpu_function_list = open_file(framework, "tpu", 2, tpu_function_list, soft)
    tpu_function_list = open_file(framework, "tpu", 3, tpu_function_list, soft)
    gpu_function_list = open_file(framework, "gpu", 1, {}, soft)
    gpu_function_list = open_file(framework, "gpu", 2, gpu_function_list, soft)
    gpu_function_list = open_file(framework, "gpu", 3, gpu_function_list, soft)
    function_keys = gpu_function_list.keys()
    return gpu_function_list, tpu_function_list, function_keys


palette_tab10 = sns.color_palette("tab10", 10)
framework = "tensorflow"
frameworkTitle = framework.capitalize()
soft_gpu_function_list, soft_tpu_function_list, soft_function_keys = fetch_data(
    framework, True
)
gpu_function_list, tpu_function_list, function_keys = fetch_data(framework, False)
data = {"Device": [], "Time": []}


def make_chart(non_soft, soft, function_keys, device):
    for key in function_keys:
        if len(non_soft[key]["operations"]) == len(soft[key]["operations"]):
            for operation in non_soft[key]["operations"]:
                # print("KEPT KEY", key)
                data["Device"].append("No Soft Placement")
                data["Time"].append(operation * 1e9)

            for operation in soft[key]["operations"]:
                # print("KEPT KEY", key)
                data["Device"].append("Soft Placement")
                data["Time"].append(operation * 1e9)
        else:
            print(
                "NON MATCHING KEYS",
                key,
                len(non_soft[key]["operations"]),
                len(soft[key]["operations"]),
            )

    sns.set(font_scale=15)
    # Create a Pandas DataFrame
    df = pd.DataFrame(data)
    f, ax = plt.subplots(figsize=(68, 75))

    # ax.set(xlim=(0, 500000), xticks=np.arange(0, 500000, 1000))

    # for i in ax.containers:
    #     ax.bar_label(i, rotation=45)
    # plt.xticks(rotation=45)

    # Create the Seaborn plot
    # plt.figure(figsize=(10, 6))
    # kdeplot = sns.kdeplot(data=df, x="Time", hue="Device", multiple="stack", cut=0)
    sns.histplot(
        data=df,
        x="Time",
        hue="Device",
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
        binwidth=1000,
        log_scale=True,
        palette=palette_tab10,
    )

    l1 = ax.lines[0]
    l2 = ax.lines[1]
    x1 = l1.get_xydata()[:, 0]
    y1 = l1.get_xydata()[:, 1]
    x2 = l2.get_xydata()[:, 0]
    y2 = l2.get_xydata()[:, 1]

    ax.set_xticks([1e4, 1e6, 1e8, 1e10, 1e12])
    ax.fill_between(x1, y1, color=palette_tab10[1], alpha=0.3)
    ax.fill_between(x2, y2, color=palette_tab10[0], alpha=0.3)

    # kdeplot = sns.boxplot(x='Function', y='Time', data=df)

    # print("X TICKS", kdeplot.get_)
    # kdeplot.set_xticklabels(kdeplot.get_xticklabels(), size = 100)
    # kdeplot.set_yticklabels(kdeplot.get_yticks(), size = 85)
    # Customize the plot
    plt.xlabel("Time (ns)", labelpad=55)
    plt.ylabel("Density", labelpad=55)
    # plt.xticks(rotation=90, pad=20)

    plt.title(device + " Soft Placement Comparison", pad=100)
    ax.tick_params(axis="x", pad=35, labelsize=150)
    ax.tick_params(axis="y", pad=45, labelsize=150)

    plt.savefig(
        f"plot_images/soft_placement_comparison/soft_{device}_density_plot.png",
        bbox_inches="tight",
    )
    plt.show()
    # break
    # Show the plot


make_chart(tpu_function_list, soft_tpu_function_list, function_keys, "TPU")
make_chart(gpu_function_list, soft_gpu_function_list, function_keys, "GPU")
