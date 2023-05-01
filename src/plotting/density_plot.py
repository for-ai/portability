import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import numpy as np


# for key in function_keys:
#      if len(gpu_function_list[key]["operations"]) != len(cpu_vm_tpu_function_list[key]["operations"]) or len(tpu_function_list[key]["operations"]) != len(cpu_vm_tpu_function_list[key]["operations"]) or len(gpu_function_list[key]["operations"]) != len(tpu_function_list[key]["operations"]):
#         print("NON MATCHING KEYS", key, len(cpu_vm_tpu_function_list[key]["operations"]), len(gpu_function_list[key]["operations"]), len(tpu_function_list[key]["operations"]))
# function_keys = set(function_keys.map(lambda x: x.split(":")[0]))

palette_tab10 = sns.color_palette("tab10", 10)
framework = "tensorflow"
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


sns.set(font_scale=15)
# Create a Pandas DataFrame
df = pd.DataFrame(data)
f, ax = plt.subplots(figsize=(58, 55))


# ax.set(xlim=(0, 500000), xticks=np.arange(0, 500000, 1000))

# for i in ax.containers:
#     ax.bar_label(i, rotation=45)
# plt.xticks(rotation=45)


# Create the Seaborn plot
# plt.figure(figsize=(10, 6))
# kdeplot = sns.kdeplot(data=df, x="Time", hue="Device", multiple="stack", cut=0)
sns.histplot(data=df, x="Time", hue="Device", kde=True, element="step", stat="density", common_norm=False, binwidth=1000, log_scale=True, palette=palette_tab10)

l1 = ax.lines[0]
l2 = ax.lines[1]
x1 = l1.get_xydata()[:,0]
y1 = l1.get_xydata()[:,1]
x2 = l2.get_xydata()[:,0]
y2 = l2.get_xydata()[:,1]


ax.fill_between(x1,y1, color=palette_tab10[1], alpha=0.3)
ax.fill_between(x2,y2, color=palette_tab10[0], alpha=0.3)



# kdeplot = sns.boxplot(x='Function', y='Time', data=df)

# print("X TICKS", kdeplot.get_)
# kdeplot.set_xticklabels(kdeplot.get_xticklabels(), size = 100)
# kdeplot.set_yticklabels(kdeplot.get_yticks(), size = 85)
# Customize the plot
plt.xlabel('Time (ns)', labelpad=55)
plt.ylabel('Density', labelpad=55)
plt.title(f'{frameworkTitle} Density', pad=35)
# plt.xticks(rotation=90, pad=20)
ax.tick_params(axis='x', pad=35, labelsize=150)
ax.tick_params(axis='y', pad=45, labelsize=150)




plt.savefig(f'plot_images/{framework}_density_plot.png', bbox_inches='tight')
plt.show()
# break
# Show the plot
