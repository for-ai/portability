import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

framework = "tensorflow"
f = open("../" + framework + '_tpu.json')
function_list = json.load(f)
function_list = function_list.items()
f.close()

data = {'Function': [],
        'Time': []}

for key, value in function_list:
    for operation in value["operations"]:
        # if "cpu" in key:
            print("KEPT KEY", key)
            data['Function'].append(key.split(":")[0])
            data['Time'].append(operation * 1000)
        # else:
        #     print("REMOVED KEY", key)


# Create a Pandas DataFrame
df = pd.DataFrame(data)
f, ax = plt.subplots(figsize=(50, 15))

ax.set(yscale="log", ylim=(10e-4, 10000), xlabel="Function Name",
       ylabel="Time taken for " + framework + " on TPU")

for i in ax.containers:
    ax.bar_label(i, rotation=45)
plt.xticks(rotation=45)


# Create the Seaborn plot
# plt.figure(figsize=(10, 6))
sns.boxplot(x='Function', y='Time', data=df)

# Customize the plot
plt.title('Execution Time of Tensorflow Functions (TPU)')
plt.xlabel('Function')
plt.ylabel('Time (milliseconds)')

plt.savefig(framework + '_tpu_log_plot.png')
# Show the plot
plt.show()