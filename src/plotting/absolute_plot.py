import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import utils

device = "tpu"
framework = "tensorflow"
frameworkTitle = framework.capitalize()
gpu_function_list, tpu_function_list, function_keys = utils.fetch_data(framework)

if device == "tpu":
    function_list = tpu_function_list
elif device == "gpu":
    function_list = gpu_function_list

data = {'Function': [], 'Time': []}
for key in function_keys:
    # if key == "Dataset_test.py":
    #     continue

    # if len(gpu_function_list[key]["operations"]) != len(tpu_function_list[key]["operations"]):
    #     continue
    for operation in function_list[key]["operations"]:
            # print("KEPT KEY", key)
            data['Function'].append(key)
            # cpu_baseline = gpu_function_list[key]["operations"][i]
            # percent_difference = ((operation - cpu_baseline)/cpu_baseline)*100
            data['Time'].append(operation)

sns.set(font_scale=15)
# Create a Pandas DataFrame
df = pd.DataFrame(data)
f, ax = plt.subplots(figsize=(225, 100))


ax.set(yscale="log", ylim=(10e-7, 100), xlabel="Function Name",
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
