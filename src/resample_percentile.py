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


framework = "tensorflow"
decile_to_choose = 100
f = open(framework + '_frequencies.json')
function_list = json.load(f)
function_list = function_list.items()
f.close()

sample = True
length = len(function_list)
sampled_list = []

# code.interact(local=dict(globals(), **locals()))
# buckets = pd.qcut(df["frequency"], 10, labels=df["function"])
# code.interact(local=dict(globals(), **locals()))


od = collections.OrderedDict(
    sorted(function_list, key=lambda x: x[1]), reverse=True)
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
for function_name, amount in dict_items:
    if amount > current_rank:
        current_decile += 10
        current_rank = np.percentile(frequencies, current_decile)
    if current_decile not in percentile_buckets:
        percentile_buckets[current_decile] = []
    percentile_buckets[current_decile].append(
        (function_name, amount))

print(percentile_buckets)
sample = random.sample(percentile_buckets[decile_to_choose], 1)
print(sample)
