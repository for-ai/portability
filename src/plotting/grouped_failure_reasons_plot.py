import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

# "Gradcheck Failure",
#
#         "Type Failure",
#         "Function Not Implemented",
#         "Timeout",
#         "Memory Access Issue",

# "Zero Sized Variable Failure",
#         "Function Not Implemented or Partially Implemented",
#         "assert_like Failure",
#         "Type Failure",
#         "Crossing the XLA/TF Boundary",


# "Function Not Implemented or Partially Implemented or Sparse Tensor Failure", "zero sized variable failure",
# "Type Failure"
# "Float Precision Error", "Gradcheck Failure"
# "Timeout",
# "Memory Access Issue",

# Assuming you have a DataFrame named 'data' with columns 'failure_category' and 'failure_reason'
torch_tpu_data = {
    "failure_category": [
        "Not Implemented",
        "Type Failure",
        "Timeout",
        "Memory Issue",
        "Float Precision Error",
    ],
    "count": [9, 7, 1, 3, 6],
}
torch_gpu_data = {
    "failure_category": [
        "Not Implemented",
        "Type Failure",
        "Timeout",
        "Memory Issue",
        "Float Precision Error",
    ],
    "count": [1, 1, 0, 1, 0],
}

tensorflow_tpu_data = {
    "failure_category": [
        "Not Implemented",
        "Type Failure",
        "Timeout",
        "Memory Issue",
        "Float Precision Error",
    ],
    "count": [13, 2, 0, 1, 4],
}

tensorflow_gpu_data = {
    "failure_category": [
        "Not Implemented",
        "Type Failure",
        "Timeout",
        "Memory Issue",
        "Float Precision Error",
    ],
    "count": [13, 0, 0, 1, 0],
}


jax_tpu_data = {
    "failure_category": [
        "Not Implemented",
        "Type Failure",
        "Timeout",
        "Memory Issue",
        "Float Precision Error",
    ],
    "count": [2, 0, 0, 0, 0],
}

jax_gpu_data = {
    "failure_category": [
        "Not Implemented",
        "Type Failure",
        "Timeout",
        "Memory Issue",
        "Float Precision Error",
    ],
    "count": [1, 0, 0, 0, 0],
}


def make_percentages(data):
    total = sum(data["count"])
    for index, row in enumerate(data["count"]):
        data["count"][index] = (row / total) * 100
    return data


torch_tpu_data = pd.DataFrame(
    make_percentages(torch_tpu_data)
)  # .sort_values("count", ascending=False)
torch_gpu_data = pd.DataFrame(
    make_percentages(torch_gpu_data)
)  # .sort_values("count", ascending=False)
tensorflow_tpu_data = pd.DataFrame(
    make_percentages(tensorflow_tpu_data)
)  # .sort_values(
tensorflow_gpu_data = pd.DataFrame(
    make_percentages(tensorflow_gpu_data)
)  # .sort_values(
# "count", ascending=False
# )
jax_tpu_data = pd.DataFrame(
    make_percentages(jax_tpu_data)
)  # .sort_values("count", ascending=False)
jax_gpu_data = pd.DataFrame(
    make_percentages(jax_gpu_data)
)  # .sort_values("count", ascending=False)
# rotation_angle = 45


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(text, width=width, break_long_words=break_long_words)
        )
    ax.set_xticklabels(labels, rotation=0)


# Create a grouped bar chart with Seaborn
def create_plot(data, title):
    sns.set(font_scale=3.0)
    f, ax = plt.subplots(figsize=(15, 6))
    ax.set(ylim=(0, 100))
    sns.barplot(data=data, x="failure_category", y="count")
    ax.xaxis.labelpad = 25
    ax.yaxis.labelpad = 25
    wrap_labels(ax, 6)
    # xticks = plt.xticks(rotation=rotation_angle, ha="right")
    # for tick in xticks[
    #     1
    # ]:  # xtis[1] contains the list of Text objects for the x-axis labels
    #     tick.set_position((tick.get_position()[0] - 0.25, tick.get_position()[1]))

    # Customize the chart
    plt.xlabel("")
    plt.ylabel("Percentage")

    # Show the chart
    plt.savefig(f"plot_images/failure_category/{title}.png", bbox_inches="tight")
    # plt.show()


create_plot(torch_tpu_data, "torch_tpu_categories")
create_plot(torch_gpu_data, "torch_gpu_categories")
# create_plot(tensorflow_tpu_data, "tensorflow_tpu_categories")
create_plot(tensorflow_tpu_data, "tensorflow_tpu_categories")
create_plot(tensorflow_gpu_data, "tensorflow_gpu_categories")
create_plot(jax_tpu_data, "jax_tpu_categories")
create_plot(jax_gpu_data, "jax_gpu_categories")
