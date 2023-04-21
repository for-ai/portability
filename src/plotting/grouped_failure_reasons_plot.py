import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'data' with columns 'failure_category' and 'failure_reason'
torch_tpu_data = {'failure_category': ["Gradcheck Failure", "Spare Tensor Failure", "Type Failure", "Function Not Implemented", "Timeout", "Memory Access Issue"], 'count': [5, 2, 8, 6, 1, 3]}
torch_gpu_data = {'failure_category': ["Gradcheck Failure", "Spare Tensor Failure", "Type Failure", "Function Not Implemented", "Timeout", "Memory Access Issue"], 'count': [1, 0, 1, 1, 0, 0]}

tensorflow_tpu_data= {"failure_category": ["Zero Sized Variable Failure", "Function Not Implemented", "assert_like Failure", "Type Failure", "Crossing the XLA/TF Boundary"], "count": [1, 5, 1, 4, 3]}

# Create a grouped bar chart with Seaborn
f, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=torch_tpu_data, x='failure_category', y='count')
ax.tick_params(axis='x', rotation=90)

# Customize the chart
plt.title('Count of Failure Reasons by Failure Category')
plt.xlabel('Failure Category')
plt.ylabel('Count')

# Show the chart
plt.savefig('plot_images/torch_tpu_categories.png', bbox_inches='tight')
plt.show()

f, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=torch_gpu_data, x='failure_category', y='count')
ax.tick_params(axis='x', rotation=90)

# Customize the chart
plt.title('Count of Failure Reasons by Failure Category')
plt.xlabel('Failure Category')
plt.ylabel('Count')

# Show the chart
plt.savefig('plot_images/torch_gpu_categories.png', bbox_inches='tight')
plt.show()

f, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=tensorflow_tpu_data, x='failure_category', y='count')
ax.tick_params(axis='x', rotation=90)

# Customize the chart
plt.title('Count of Failure Reasons by Failure Category')
plt.xlabel('Failure Category')
plt.ylabel('Count')

# Show the chart
plt.savefig('plot_images/tensorflow_gpu_categories.png', bbox_inches='tight')
plt.show()