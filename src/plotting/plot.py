import code
import enum
import json
import numpy as np
import collections
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
import csv
import numpy as np
import math

sns.set_theme(style="whitegrid")


# def percentile_rank(length, current, previous):
#     return ((previous + (0.5 * current)) / length) * 100


framework = "jax"
f = open(framework + "_frequencies.json")
function_list = json.load(f)
function_list = function_list.items()
f.close()

sample = True
length = len(function_list)
sampled_list = []

# code.interact(local=dict(globals(), **locals()))
# buckets = pd.qcut(df["frequency"], 10, labels=df["function"])
# code.interact(local=dict(globals(), **locals()))


if sample:
    od = collections.OrderedDict(
        sorted(function_list, key=lambda x: x[1]), reverse=True
    )
    frequency_amounts = {}
    last_prior = 0
    current_prior = 0
    dict_items = list(sorted(list(od.items()), key=lambda x: x[1]))
    functions = [key for key, value in dict_items]
    frequencies = np.array([value for key, value in dict_items])
    # current_value = dict_items[0][1]
    # for function, frequency in dict_items:
    #     if frequency != current_value:
    #         last_prior = current_prior
    #         current_value = frequency
    #     if frequency not in frequency_amounts:
    #         frequency_amounts[frequency] = {
    #             "amount": 0,
    #             "prior_amount": last_prior
    #         }
    #     frequency_amounts[frequency]["amount"] += 1
    #     current_prior += 1
    current_decile = 10
    percentile_buckets = {}
    last_rank = 0
    current_rank = np.percentile(frequencies, current_decile)
    for function_name, amount in dict_items:
        if amount > current_rank:
            current_decile += 10
            current_rank = np.percentile(frequencies, current_decile)
        if current_decile not in percentile_buckets:
            percentile_buckets[current_decile] = []
        percentile_buckets[current_decile].append((function_name, amount))
    # for i in range(1, 11):
    #     decile = math.ceil(i * 10 / 100 * length)

    #     print("INDICES", previous_decile, decile)
    #     if i == 0 or i == 1:
    #         sampled_list += list(od.items())[previous_decile:decile][0:10]
    #     else:
    #         sampled_list += sorted(random.sample(list(od.items())
    #                                              [previous_decile:decile], 10), key=lambda x: x[1], reverse=True)
    #     previous_decile = decile
    import code

    code.interact(local=dict(globals(), **locals()))
    for bucket in percentile_buckets:
        # if bucket == 100:
        #     sampled_list += percentile_buckets[bucket][-20:]
        # else:
        sampled_list += random.sample(percentile_buckets[bucket], 5)
    function_list = sampled_list

od = collections.OrderedDict(sorted(function_list, key=lambda x: x[1], reverse=True))

keys = od.keys()
y = [value for value in list(od.values())]
# plt.figure(figsize=(9, 3))
df = pd.DataFrame(data={"function": list(keys), "frequency": y})
f, ax = plt.subplots(figsize=(250, 20))

sns.set_color_codes("pastel")
sns.barplot(x="function", y="frequency", data=df)

ax.set(
    ylim=(0, 7000),
    xlabel="",
    ylabel="Funtion Frequency in " + framework + " for 3000 files",
)

sns.despine(left=True, bottom=True)

# fig, ax = plt.subplots(figsize=(300, 100))
# ax.bar(keys, y, width=0.35, edgecolor="white", linewidth=0.8)
# ax.set(xlim=(0, 3000), xticks=np.arange(1, 3000),
#        ylim=(0, 1000), yticks=np.arange(1, 20) * 50)

for i in ax.containers:
    ax.bar_label(i, rotation=90)
plt.xticks(rotation=90)
plt.savefig(framework + "_plot.png")

# open the file in the write mode
# f = open(framework + '_sampled_datapoints.csv', 'w')

# # create the csv writer
# writer = csv.writer(f)
# writer.writerow(["function", "frequency", "percentile"])
# code.interact(local=dict(globals(), **locals()))
# for key, value in list(od.items()):
#     writer.writerow([key, value[0], value[1]])

# # close the file
# f.close()
od = collections.OrderedDict(reversed(list(od.items())))

f = open(framework + "_sampled_datapoints.csv", "w")

code.interact(local=dict(globals(), **locals()))
# create the csv writer
writer = csv.writer(f)
writer.writerow(["function", "frequency", "bucket"])
for index, (key, value) in enumerate(list(od.items())):
    # if index >= len(od) - 20:
    #     writer.writerow([key, value, "Top 20"])
    # else:
    bucket = math.ceil((index + 1) / 10) * 10
    writer.writerow([key, value, bucket])

# close the file
f.close()
