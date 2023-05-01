import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'data' with columns 'failure_category' and 'failure_reason'
torch_tpu_data = {'failure_category': ["Gradcheck Failure", "Spare Tensor Failure", "Type Failure", "Function Not Implemented", "Timeout", "Memory Access Issue"], 'count': [5, 2, 8, 6, 1, 3]}
torch_gpu_data = {'failure_category': ["Gradcheck Failure", "Spare Tensor Failure", "Type Failure", "Function Not Implemented", "Timeout", "Memory Access Issue"], 'count': [1, 0, 1, 1, 0, 0]}

tensorflow_tpu_data= {"failure_category": ["Zero Sized Variable Failure", "Function Not Implemented", "assert_like Failure", "Type Failure", "Crossing the XLA/TF Boundary"], "count": [1, 5, 1, 4, 3]}
sns.set(font_scale=1.5)


torch_tpu_data = pd.DataFrame(torch_tpu_data).sort_values('count', ascending=False)
torch_gpu_data = pd.DataFrame(torch_gpu_data).sort_values('count', ascending=False)
tensorflow_tpu_data = pd.DataFrame(tensorflow_tpu_data).sort_values('count', ascending=False)

rotation_angle = 45

# Create a grouped bar chart with Seaborn
f, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=torch_tpu_data, x='failure_category', y='count')
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 25
xticks = plt.xticks(rotation=rotation_angle, ha='right')
for tick in xticks[1]:  # xticks[1] contains the list of Text objects for the x-axis labels
    tick.set_position((tick.get_position()[0] - 0.25, tick.get_position()[1]))


# Customize the chart
plt.title('Count of Failure Reasons by Failure Category', pad=20)
plt.xlabel('Failure Category')
plt.ylabel('Count')

# Show the chart
plt.savefig('plot_images/failure_category/torch_tpu_categories.png', bbox_inches='tight')
plt.show()

f, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=torch_gpu_data, x='failure_category', y='count')
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 25
xticks = plt.xticks(rotation=rotation_angle, ha='right')
for tick in xticks[1]:  # xticks[1] contains the list of Text objects for the x-axis labels
    tick.set_position((tick.get_position()[0] - 0.25, tick.get_position()[1]))


# Customize the chart
plt.title('Count of Failure Reasons by Failure Category', pad=20)
plt.xlabel('Failure Category')
plt.ylabel('Count')

# Show the chart
plt.savefig('plot_images/failure_category/torch_gpu_categories.png', bbox_inches='tight')
plt.show()

f, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=tensorflow_tpu_data, x='failure_category', y='count')
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 25
xticks = plt.xticks(rotation=rotation_angle, ha='right')
for tick in xticks[1]:  # xticks[1] contains the list of Text objects for the x-axis labels
    tick.set_position((tick.get_position()[0] - 0.25, tick.get_position()[1]))


# Customize the chart
plt.title('Count of Failure Reasons by Failure Category', pad=20)
plt.xlabel('Failure Category')
plt.ylabel('Count')

# Show the chart
plt.savefig('plot_images/failure_category/tensorflow_tpu_categories.png', bbox_inches='tight')
plt.show()