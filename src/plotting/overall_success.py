import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import utils
import numpy as np

gpu_function_list, tpu_function_list, function_keys = utils.fetch_data(
    "tensorflow", by_function=True, consolidation_method="mean"
)
percent_count = 0
total_count = 0


palette_tab10 = sns.color_palette("tab10", 10)


def show_plot(data, framework):
    # Create a grouped bar chart with Seaborn
    f, ax = plt.subplots(figsize=(35, 15))

    sns.set(font_scale=10)
    sns.barplot(data=data, x="Category", y="Percent", palette=palette_tab10)
    ax.tick_params(axis="x", pad=100)
    ax.xaxis.labelpad = 25
    ax.yaxis.labelpad = 25

    # xticks = plt.xticks(rotation=rotation_angle, ha='right')
    # for tick in xticks[1]:  # xticks[1] contains the list of Text objects for the x-axis labels
    #     tick.set_position((tick.get_position()[0] - 0.25, tick.get_position()[1]))

    # Customize the chart
    plt.ylabel("Percent")

    # Show the chart
    plt.savefig(f"plot_images/overall_success.png", bbox_inches="tight")
    plt.show()


def get_percent_count(gpu_function_list, tpu_function_list, function_keys):
    percent_count = 0
    total_count = 0
    for key in function_keys:
        # if key == "Dataset_test.py":
        #     continue

        # if len(gpu_function_list[key]["operations"]) != len(tpu_function_list[key]["operations"]):
        #     print("NON MATCHING", key, len(gpu_function_list[key]["operations"]), len(tpu_function_list[key]["operations"]))
        #     continue

        if key in tpu_function_list and key in gpu_function_list:
            tpu_operations = tpu_function_list[key]["operations"]
            gpu_operations = gpu_function_list[key]["operations"]
            if len(tpu_operations) == 0:
                tpu_ratio = 0
            else:
                tpu_ratio = sum(tpu_operations) / len(tpu_operations)

            if len(gpu_operations) == 0:
                gpu_ratio = 0
            else:
                gpu_ratio = sum(gpu_operations) / len(gpu_operations)

            print(key, tpu_ratio, gpu_ratio, len(tpu_operations), len(gpu_operations))
            if np.mean(tpu_operations, dtype=np.float64) < np.mean(
                gpu_operations, dtype=np.float64
            ):
                percent_count += 1
            total_count += 1
    return percent_count, total_count


tensorflow_faster_count, tensorflow_total_count = get_percent_count(
    gpu_function_list, tpu_function_list, function_keys
)

tensorflow_percentage = tensorflow_faster_count / tensorflow_total_count * 100


gpu_function_list, tpu_function_list, function_keys = utils.fetch_data("torch")

torch_faster_count, torch_total_count = get_percent_count(
    gpu_function_list, tpu_function_list, function_keys
)
torch_percentage = torch_faster_count / torch_total_count * 100

gpu_function_list, tpu_function_list, function_keys = utils.fetch_data("jax")


jax_faster_count, jax_total_count = get_percent_count(
    gpu_function_list, tpu_function_list, function_keys
)
jax_percentage = jax_faster_count / jax_total_count * 100


print("jax_percentage", 100 - jax_percentage)
print(
    "Tensorflow percentage",
    100 - tensorflow_percentage,
    tensorflow_total_count - tensorflow_faster_count,
)
print(
    "Torch percentage", 100 - torch_percentage, torch_total_count - torch_faster_count
)


framework = "torch"
# # Assuming you have a DataFrame named 'data' with columns 'failure_category' and 'failure_reason'
torch_data = {
    "Category": [f"All {framework}", "Ran Successfully", "Ran Faster on TPU"],
    "Percent": [100, 54.69, torch_percentage],
}
framework = "tensorflow"
tensorflow_data = {
    "Category": [f"All {framework}", "Ran Successfully", "Ran Faster on TPU"],
    "Percent": [100, 66.15, tensorflow_percentage],
}
framework = "jax"
jax_data = {
    "Category": [f"All {framework}", "Ran Successfully", "Ran Faster on TPU"],
    "Percent": [100, 96.88, jax_percentage],
}

overall = {
    "Category": ["TensorFlow", "PyTorch", "JAX"],
    "Percent": [tensorflow_percentage, torch_percentage, jax_percentage],
}

sns.set(font_scale=5)

tensorflow_data = pd.DataFrame(tensorflow_data)
torch_data = pd.DataFrame(torch_data)
jax_data = pd.DataFrame(jax_data)

# show_plot(tensorflow_data, "tensorflow")
# show_plot(torch_data, "torch")
# show_plot(jax_data, "jax")

show_plot(overall, "Percentage Faster on TPU")
# rotation_angle = 45
# palette_tab10 = sns.color_palette("tab10", 10)

# # Create a grouped bar chart with Seaborn
# f, ax = plt.subplots(figsize=(20, 22))
# sns.barplot(data=torch_data, x="Category", y="Percent", palette=palette_tab10)
# ax.tick_params(axis="x", pad=100, rotation=90)
# ax.xaxis.labelpad = 25
# ax.yaxis.labelpad = 25
# palette_tab10 = sns.color_palette("tab10", 10)

# # xticks = plt.xticks(rotation=rotation_angle, ha='right')
# # for tick in xticks[1]:  # xticks[1] contains the list of Text objects for the x-axis labels
# #     tick.set_position((tick.get_position()[0] - 0.25, tick.get_position()[1]))


# # Customize the chart
# plt.title("Overall Torch Success", pad=20)
# plt.ylabel("Percent")

# # Show the chart
# plt.savefig("plot_images/overall_success/torch.png", bbox_inches="tight")
# plt.show()

# sns.set(font_scale=5)
# f, ax = plt.subplots(figsize=(20, 22))
# sns.barplot(data=tensorflow_data, x="Category", y="Percent", palette=palette_tab10)
# ax.tick_params(axis="x", pad=100, rotation=90)
# ax.xaxis.labelpad = 25
# ax.yaxis.labelpad = 25

# # xticks = plt.xticks(rotation=rotation_angle, ha='right')
# # for tick in xticks[1]:  # xticks[1] contains the list of Text objects for the x-axis labels
# #     tick.set_position((tick.get_position()[0] - 0.25, tick.get_position()[1]))


# # Customize the chart
# plt.title("Overall Tensorflow Success", pad=20)
# plt.ylabel("Percent")

# # Show the chart
# plt.savefig("plot_images/overall_success/tensorflow.png", bbox_inches="tight")
# plt.show()
