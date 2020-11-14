import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()
sns.set_style("whitegrid")

data = np.load("affectations.npy")

def filter_func(item):
    return len(np.unique(item)) == len(item)


filter_iterable = filter(filter_func, data)
filtered_data = np.array([item for item in filter_iterable])

#filtered_data = data

prepared_data = []

for d in filtered_data:
    for i in range(filtered_data.shape[1]):
        prepared_data.append(["Base {}".format(i), "Agent {}".format(d[i])])

"""
for base in range(filtered_data.shape[1]):
    base_data = []

    for agent in range(filtered_data.shape[1]):
        count = len(np.where(filtered_data[:, base] == agent)[0])
        base_data.append(count)

    prepared_data.append(base_data)
"""


dataframe = pd.DataFrame(prepared_data, columns=['Bases', 'Agents'])
ax = sns.countplot(x="Bases", hue="Agents", data=dataframe)
plt.savefig("distribution.png")
plt.show()