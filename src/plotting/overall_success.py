import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import utils

gpu_function_list, tpu_function_list, function_keys = utils.fetch_data("tensorflow")
percent_count = 0
total_count = 0

for key in function_keys:
    if key == "Dataset_test.py":
        continue

    # if len(gpu_function_list[key]["operations"]) != len(tpu_function_list[key]["operations"]):
    #     print("NON MATCHING", key, len(gpu_function_list[key]["operations"]), len(tpu_function_list[key]["operations"]))
    #     continue

    tpu_operations = tpu_function_list[key]['operations']
    gpu_operations = gpu_function_list[key]['operations']
    if len(tpu_operations) > 0 and (sum(tpu_operations) / len(tpu_operations)) <= (sum(gpu_operations) / len(gpu_operations)):
        percent_count += 1
    total_count += 1


tensorflow_percentage = percent_count / total_count * 100


gpu_function_list, tpu_function_list, function_keys = utils.fetch_data("torch")

percent_count = 0
total_count = 0
for key in function_keys:
    if key == "Dataset_test.py":
        continue

    # if len(gpu_function_list[key]["operations"]) != len(tpu_function_list[key]["operations"]):
    #     print("NON MATCHING", key, len(gpu_function_list[key]["operations"]), len(tpu_function_list[key]["operations"]))
    #     continue

    tpu_operations = tpu_function_list[key]['operations']
    gpu_operations = gpu_function_list[key]['operations']
    if len(tpu_operations) > 0 and sum(tpu_operations) / len(tpu_operations) <= sum(gpu_operations) / len(gpu_operations):
        percent_count += 1
    total_count += 1


torch_percentage = percent_count / total_count * 100


print("Tensorflow percentage", tensorflow_percentage)
print("Torch percentage", torch_percentage)


framework = "torch"
# Assuming you have a DataFrame named 'data' with columns 'failure_category' and 'failure_reason'
torch_data = {'Category': [f'All {framework}', "Ran Successfully", "Ran Faster on TPU"], 'Percent': [100, 54.69, torch_percentage]}
framework = "tensorflow"
tensorflow_data = {'Category': [f'All {framework}', "Ran Successfully", "Ran Faster on TPU"], 'Percent': [100, 66.15, tensorflow_percentage]}

sns.set(font_scale=5)

tensorflow_data = pd.DataFrame(tensorflow_data)
torch_data = pd.DataFrame(torch_data)

rotation_angle = 45
palette_tab10 = sns.color_palette("tab10", 10)

# Create a grouped bar chart with Seaborn
f, ax = plt.subplots(figsize=(20, 22))
sns.barplot(data=torch_data, x='Category', y='Percent', palette=palette_tab10)
ax.tick_params(axis='x', pad=100, rotation=90)
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 25
palette_tab10 = sns.color_palette("tab10", 10)

# xticks = plt.xticks(rotation=rotation_angle, ha='right')
# for tick in xticks[1]:  # xticks[1] contains the list of Text objects for the x-axis labels
#     tick.set_position((tick.get_position()[0] - 0.25, tick.get_position()[1]))


# Customize the chart
plt.title('Overall Torch Success', pad=20)
plt.ylabel('Percent')

# Show the chart
plt.savefig('plot_images/overall_success/torch.png', bbox_inches='tight')
plt.show()

sns.set(font_scale=5)
f, ax = plt.subplots(figsize=(20, 22))
sns.barplot(data=tensorflow_data, x='Category', y='Percent', palette=palette_tab10)
ax.tick_params(axis='x', pad=100, rotation=90)
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 25

# xticks = plt.xticks(rotation=rotation_angle, ha='right')
# for tick in xticks[1]:  # xticks[1] contains the list of Text objects for the x-axis labels
#     tick.set_position((tick.get_position()[0] - 0.25, tick.get_position()[1]))


# Customize the chart
plt.title('Overall Tensorflow Success', pad=20)
plt.ylabel('Percent')

# Show the chart
plt.savefig('plot_images/overall_success/tensorflow.png', bbox_inches='tight')
plt.show()
