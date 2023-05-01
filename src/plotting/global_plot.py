import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import utils

framework = "tensorflow"
frameworkTitle = framework.capitalize()
gpu_function_list, tpu_function_list, function_keys = utils.fetch_data(framework)

# f = open("./tensorflow_timings/" + 'gpu.json')
# gpu_function_list = combine_function_tests(json.load(f))
# f.close()

# function_keys = function_list.keys()

# for key in function_keys:
#      if len(gpu_function_list[key]["operations"]) != len(cpu_vm_tpu_function_list[key]["operations"]) or len(tpu_function_list[key]["operations"]) != len(cpu_vm_tpu_function_list[key]["operations"]) or len(gpu_function_list[key]["operations"]) != len(tpu_function_list[key]["operations"]):
#         print("NON MATCHING KEYS", key, len(cpu_vm_tpu_function_list[key]["operations"]), len(gpu_function_list[key]["operations"]), len(tpu_function_list[key]["operations"]))
# function_keys = set(function_keys.map(lambda x: x.split(":")[0]))
data = {'Device': [], 'Time': []}
for key, item in tpu_function_list.items():
	for operation in item["operations"]:
		data["Device"].append("TPU")
		data["Time"].append(operation)

for key, item in gpu_function_list.items():
	for operation in item["operations"]:
		data["Device"].append("GPU")
		data["Time"].append(operation)


sns.set(font_scale=5)
# Create a Pandas DataFrame
df = pd.DataFrame(data)
f, ax = plt.subplots(figsize=(30, 20))


ax.set(yscale="log", ylim=(10e-8, 100), xlabel="Device",
    ylabel="Time taken (milliseconds)")

# for i in ax.containers:
#     ax.bar_label(i, rotation=45)
# plt.xticks(rotation=45)


# Create the Seaborn plot
# plt.figure(figsize=(10, 6))
box_plot = sns.boxplot(x='Device', y='Time', data=df)

# print("X TICKS", box_plot.get_)
# box_plot.set_xticklabels(box_plot.get_xticklabels(), size = 100)
# box_plot.set_yticklabels(box_plot.get_yticks(), size = 85)
# Customize the plot
plt.title(frameworkTitle + " Timings", pad=20)
plt.xlabel('Device', labelpad=20)
plt.ylabel('Time (milliseconds)', labelpad=20)
# plt.xticks(rotation=90, pad=20)
ax.tick_params(axis='x', pad=20, labelsize=50)
ax.tick_params(axis='y', pad=20, labelsize=50)




plt.savefig('plot_images/global_plot/' + framework + '_times.png')
plt.show()
# break
# Show the plot
