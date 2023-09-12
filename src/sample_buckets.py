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
import math

sns.set_theme(style="whitegrid")


# def percentile_rank(length, current, previous):
#     return ((previous + (0.5 * current)) / length) * 100


# CHANGE THESE BEFORE YOU RUN
framework = "jax"
# framework = "tensorflow"
decile_to_choose = 10


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


od = collections.OrderedDict(sorted(function_list, key=lambda x: x[1]), reverse=True)
frequency_amounts = {}
last_prior = 0
current_prior = 0
dict_items = list(sorted(list(od.items()), key=lambda x: x[1]))
functions = [key for key, value in dict_items]
frequencies = np.array([value for key, value in dict_items])
current_decile = 10
percentile_buckets = {}
last_rank = 0
current_rank = np.percentile(frequencies, current_decile)


last_bucket = []
last_rank = 0

for function_name, amount in dict_items:
    if amount > current_rank:
        last_bucket = percentile_buckets[current_decile]
        last_rank = current_rank

        current_decile += 10
        current_rank = np.percentile(frequencies, current_decile)
        if current_rank == last_rank:
            percentile_buckets[current_decile] = last_bucket
            current_decile += 10
            current_rank = np.percentile(frequencies, current_decile)

    if current_decile not in percentile_buckets:
        percentile_buckets[current_decile] = []

    percentile_buckets[current_decile].append((function_name, amount))


sampled_buckets = []
for key, value in percentile_buckets.items():
    flatlist = [element for sublist in sampled_buckets for element in sublist]
    list_to_sample_from = list(set(value) - set(flatlist))
    sampled_values = random.sample(list_to_sample_from, 5)
    sampled_values = sorted(sampled_values, key=lambda x: x[1])
    sampled_buckets.append(sampled_values)


f = open(framework + "_sampled_datapoints.csv", "w")

# create the csv writer
writer = csv.writer(f)
writer.writerow(["function", "frequency", "bucket"])

for index, bucket in enumerate(sampled_buckets):
    # if index >= len(od) - 20:
    #     writer.writerow([key, value, "Top 20"])
    # else:
    for key, value in bucket:
        bucket_index = (index + 1) * 10
        writer.writerow([key, value, bucket_index])

for key, value in dict_items[-20:]:
    writer.writerow([key, value, "top 20"])

f.close()
