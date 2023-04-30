
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils


# for key in function_keys:
#      if len(gpu_function_list[key]["operations"]) != len(cpu_vm_tpu_function_list[key]["operations"]) or len(tpu_function_list[key]["operations"]) != len(cpu_vm_tpu_function_list[key]["operations"]) or len(gpu_function_list[key]["operations"]) != len(tpu_function_list[key]["operations"]):
#         print("NON MATCHING KEYS", key, len(cpu_vm_tpu_function_list[key]["operations"]), len(gpu_function_list[key]["operations"]), len(tpu_function_list[key]["operations"]))
# function_keys = set(function_keys.map(lambda x: x.split(":")[0]))

framework = "tensorflow"
gpu_function_list, tpu_function_list, function_keys = utils.fetch_data(framework)
data = {'Function': [], 'Time': []}
for key in function_keys:
    if key == "Dataset_test.py":
        continue
    i = 0

    if len(gpu_function_list[key]["operations"]) != len(tpu_function_list[key]["operations"]):
        print("NON MATCHING", key, len(gpu_function_list[key]["operations"]), len(tpu_function_list[key]["operations"]))
        continue
    for operation in tpu_function_list[key]["operations"]:
            # print("KEPT KEY", key)
            data['Function'].append(key)
            gpu_baseline = gpu_function_list[key]["operations"][i]
            percent_difference = ((operation - gpu_baseline)/gpu_baseline)*100
            data['Time'].append(percent_difference)
            i += 1

sns.set(font_scale=5)
# Create a Pandas DataFrame
df = pd.DataFrame(data)
df = df.sort_values(by='Time', ascending=False)
f, ax = plt.subplots(figsize=(56, 25))


ax.set(yscale="linear", ylim=(-200, 600), xlabel="Function Name",
    ylabel="Time taken for " + framework + " on TPU")

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
plt.title("Tensorflow TPU difference from GPU", pad=25)
plt.xlabel('Device')
plt.ylabel('Difference between GPU and TPU (percent)')
# plt.xticks(rotation=90, pad=20)
ax.tick_params(axis='x', pad=25, rotation=90, labelsize=15)
ax.tick_params(axis='y', pad=25, labelsize=15)




plt.savefig('plot_images/gpu_tpu_comparison_plot.png', bbox_inches='tight')
plt.show()
# break
# Show the plot
