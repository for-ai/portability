
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json


def combine_function_tests(list):
    functions = {}
    for key, value in list.items():
        function = key.split(":")[0]
        if not function in functions:
            functions[function] = {"operations": [], "test_time": []}
        functions[function]["operations"] += value["operations"]
        functions[function]["test_time"].append(value["test_time"])
    return functions

framework = "tensorflow"
frameworkTitle = framework.capitalize()
f = open("../" + framework + '_tpu.json')
tpu_function_list = combine_function_tests(json.load(f))
function_keys = tpu_function_list.keys()
f.close()
f = open("../" + framework + '_gpu.json')
gpu_function_list = combine_function_tests(json.load(f))
f.close()
f = open("../" + framework + '_cpu.json')
cpu_function_list = combine_function_tests(json.load(f))
f.close()


# function_keys = set(function_keys.map(lambda x: x.split(":")[0]))
for key in function_keys:
    data = {'Function': [], 'Time': []}
    i = 0
    for operation in gpu_function_list[key]["operations"]:
            # print("KEPT KEY", key)
            data['Function'].append("GPU difference from CPU")
            cpu_baseline = cpu_function_list[key]["operations"][i]
            percent_difference = ((operation - cpu_baseline)/cpu_baseline)*100
            print("percent_difference", percent_difference)
            data['Time'].append(percent_difference)
            i += 1
     
    i = 0
    for operation in tpu_function_list[key]["operations"]:
            # print("KEPT KEY", key)
            data['Function'].append("TPU difference from CPU")
            cpu_baseline = cpu_function_list[key]["operations"][i]
            percent_difference = ((operation - cpu_baseline)/cpu_baseline)*100
            data['Time'].append(percent_difference)
            i += 1

    sns.set(font_scale=2.8)
    if len(data['Function']) == 0:
        continue

    # Create a Pandas DataFrame
    df = pd.DataFrame(data)
    f, ax = plt.subplots(figsize=(25, 15))


    ax.set(yscale="symlog", ylim=(-10000, 100000), xlabel="Function Name",
        ylabel="Time taken for " + framework + " on TPU")

    # for i in ax.containers:
    #     ax.bar_label(i, rotation=45)
    # plt.xticks(rotation=45)


    # Create the Seaborn plot
    # plt.figure(figsize=(10, 6))
    sns.boxplot(x='Function', y='Time', data=df)

    # Customize the plot
    plt.title(frameworkTitle + " " + key.removesuffix("_test.py") + " Device Times")
    plt.xlabel('Device')
    plt.ylabel('Time (milliseconds)')

    plt.savefig("tensorflow_relative_plots/" + key.removesuffix("_test.py") + '_log_plot.png')
    print(key)
    plt.show()
    # break
    # Show the plot
