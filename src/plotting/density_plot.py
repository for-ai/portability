import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import numpy as np


# for key in function_keys:
#      if len(gpu_function_list[key]["operations"]) != len(cpu_vm_tpu_function_list[key]["operations"]) or len(tpu_function_list[key]["operations"]) != len(cpu_vm_tpu_function_list[key]["operations"]) or len(gpu_function_list[key]["operations"]) != len(tpu_function_list[key]["operations"]):
#         print("NON MATCHING KEYS", key, len(cpu_vm_tpu_function_list[key]["operations"]), len(gpu_function_list[key]["operations"]), len(tpu_function_list[key]["operations"]))
# function_keys = set(function_keys.map(lambda x: x.split(":")[0]))

framework = "torch"
frameworkTitle = framework.capitalize()
gpu_function_list, tpu_function_list, function_keys = utils.fetch_data(framework)
data = {'Device': [], 'Time': []}
for key in function_keys:
    if key == "Dataset_test.py":
        continue

    for operation in tpu_function_list[key]["operations"]:
            # print("KEPT KEY", key)
            data['Device'].append('TPU')
            data['Time'].append(operation * 1e9)
            print("TPU", operation * 1e9)

    for operation in gpu_function_list[key]["operations"]:
            # print("KEPT KEY", key)
            data['Device'].append('GPU')
            data['Time'].append(operation * 1e9)
            print("GPU", operation * 1e9)


sns.set(font_scale=5)
# Create a Pandas DataFrame
df = pd.DataFrame(data)
f, ax = plt.subplots(figsize=(56, 25))


# ax.set(xlim=(0, 500000), xticks=np.arange(0, 500000, 1000))

# for i in ax.containers:
#     ax.bar_label(i, rotation=45)
# plt.xticks(rotation=45)


# Create the Seaborn plot
# plt.figure(figsize=(10, 6))
# kdeplot = sns.kdeplot(data=df, x="Time", hue="Device", multiple="stack", cut=0)
sns.histplot(data=df, x="Time", hue="Device", kde=True, element="step", stat="density", common_norm=False, binwidth=1000, log_scale=True)


# kdeplot = sns.boxplot(x='Function', y='Time', data=df)

# print("X TICKS", kdeplot.get_)
# kdeplot.set_xticklabels(kdeplot.get_xticklabels(), size = 100)
# kdeplot.set_yticklabels(kdeplot.get_yticks(), size = 85)
# Customize the plot
plt.xlabel('Time (ns)', labelpad=35)
plt.ylabel('Density', labelpad=35)
plt.title(f'{frameworkTitle} Density', pad=25)
# plt.xticks(rotation=90, pad=20)
ax.tick_params(axis='x', pad=15, labelsize=50)
ax.tick_params(axis='y', pad=15, labelsize=50)




plt.savefig(f'plot_images/{framework}_density_plot.png')
plt.show()
# break
# Show the plot
