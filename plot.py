import json
import numpy as np
import collections
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme(style="whitegrid")


framework = "torch"
f = open(framework + '_frequencies.json')
function_list = json.load(f)
f.close()

length = len(function_list)
od = collections.OrderedDict(sorted(function_list.items(), key=lambda x: x[1], reverse=True))
keys = od.keys()
y = np.array(list(od.values()))
# import code; code.interact(local=dict(globals(), **locals()))
# plt.figure(figsize=(9, 3))
df = pd.DataFrame(data={"function": list(keys), "frequency": y})
f, ax = plt.subplots(figsize=(250, 20))

sns.set_color_codes("pastel")
sns.barplot(x="function", y="frequency", data=df)

ax.set(ylim=(0, 7000), xlabel="",
       ylabel="Funtion Frequency in " + framework  + " for 3000 files")

sns.despine(left=True, bottom=True)

# fig, ax = plt.subplots(figsize=(300, 100))
# ax.bar(keys, y, width=0.35, edgecolor="white", linewidth=0.8)
# ax.set(xlim=(0, 3000), xticks=np.arange(1, 3000),
#        ylim=(0, 1000), yticks=np.arange(1, 20) * 50)

for i in ax.containers:
    ax.bar_label(i, rotation=90)
plt.xticks(rotation=90)
plt.savefig(framework + '_plot.png')
