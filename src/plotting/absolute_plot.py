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

device = "tpu"
framework = "tensorflow"
frameworkTitle = framework.capitalize()
f = open("./" + framework + "_timings/" + device + "_vm/" + device + "/" + device +  '_1.json')
function_list = combine_function_tests(json.load(f))
f.close()
# f = open("./tensorflow_timings/" + 'gpu.json')
# gpu_function_list = combine_function_tests(json.load(f))
# f.close()

function_keys = function_list.keys()

# for key in function_keys:
#      if len(gpu_function_list[key]["operations"]) != len(cpu_vm_tpu_function_list[key]["operations"]) or len(tpu_function_list[key]["operations"]) != len(cpu_vm_tpu_function_list[key]["operations"]) or len(gpu_function_list[key]["operations"]) != len(tpu_function_list[key]["operations"]):
#         print("NON MATCHING KEYS", key, len(cpu_vm_tpu_function_list[key]["operations"]), len(gpu_function_list[key]["operations"]), len(tpu_function_list[key]["operations"]))
# function_keys = set(function_keys.map(lambda x: x.split(":")[0]))
data = {'Function': [], 'Time': []}
for key in function_keys:
    if key == "Dataset_test.py":
        continue

    # if len(gpu_function_list[key]["operations"]) != len(tpu_function_list[key]["operations"]):
    #     continue
    for operation in function_list[key]["operations"]:
            # print("KEPT KEY", key)
            data['Function'].append(key)
            # cpu_baseline = gpu_function_list[key]["operations"][i]
            # percent_difference = ((operation - cpu_baseline)/cpu_baseline)*100
            data['Time'].append(operation)

import code; code.interact(local=dict(globals(), **locals())) 
sns.set(font_scale=15)
# Create a Pandas DataFrame
df = pd.DataFrame(data)
f, ax = plt.subplots(figsize=(225, 100))


ax.set(yscale="log", ylim=(10e-7, 1), xlabel="Function Name",
    ylabel="Time taken for " + framework + " on " + device.upper() + " (ms)")

# for i in ax.containers:
#     ax.bar_label(i, rotation=45)
# plt.xticks(rotation=45)


# Create the Seaborn plot
# plt.figure(figsize=(10, 6))
box_plot = sns.boxplot(x='Function', y='Time', data=df)

# print("X TICKS", box_plot.get_)
# box_plot.set_xticklabels(box_plot.get_xticklabels(), size = 100)
# box_plot.set_yticklabels(box_plot.get_yticks(), size = 85)
# Customize the plot
plt.title(frameworkTitle + " " + device.upper() + " timings", pad=100)
plt.xlabel('Device')
plt.ylabel('Time (milliseconds)')
# plt.xticks(rotation=90, pad=20)
ax.tick_params(axis='x', pad=100, rotation=90, labelsize=120)
ax.tick_params(axis='y', pad=100, labelsize=120)




plt.savefig('plot_images/' + framework + "_" + device +  '_times.png', bbox_inches='tight')
plt.show()
# break
# Show the plot
