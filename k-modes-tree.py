from traceback import print_tb
import pandas as pd
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import math

import parseTree as tree

print(f'Reading csv...')
prev = pd.read_csv(r'/Users/i502765/OneDrive - Associacao Antonio Vieira/UNISINOS/Mestrado - Computação Aplicada/Practice/mestrado/data/previousStep.csv', encoding='utf-8') 
wrong = pd.read_csv(r'/Users/i502765/OneDrive - Associacao Antonio Vieira/UNISINOS/Mestrado - Computação Aplicada/Practice/mestrado/data/wrongStep.csv', encoding='utf-8') 
print('...Complete')

for index, row in prev.iterrows():
  print(row[0])
  print(tree.buildParseTree(row[0]))

equations = pd.concat([prev, wrong], axis = 1)
print(prev.describe())
print(wrong.describe())

print(type(equations))

# Clustering
for i in range(1, 6):
    nclusters = 8
    km = KModes(n_clusters=nclusters, init = "Huang", n_init = 10, verbose=0, n_jobs=4)
    cluster_labels = km.fit_predict(equations)
    equations['Cluster'] = cluster_labels

    kmodes = km.cluster_centroids_
    shape = kmodes.shape

    observations = { }
    for i in range(nclusters):
        observations[i] = list(km.labels_).count(i)
        #print("\ncluster " + str(i) +  " observations: ",list(km.labels_).count(i))

    xint = range(math.ceil(nclusters)+1)
    plt.xticks(xint)
    plt.title("Clusters distribution")
    plt.xlabel("Cluster number")
    plt.ylabel("Observations")
    plt.grid()
    plt.plot(*zip(*sorted(observations.items())))
    #plt.show()

    #clusters = []
    for n in range(nclusters):
        cluster_rows = []
        for i in range(len(km.labels_)):
            if  km.labels_[i] == n:
                cluster_rows.append(i)
            
        df = pd.DataFrame(equations.iloc[cluster_rows], columns=equations.columns)

"""         misconception = ""
        misconception1 = ""
        #print("Cluster " + str(n))
        index = 0
        for data in col:        
            if data == "RTerm1" or data == "ARTerm1":
                misconception = misconception + "= "
                misconception1 = misconception1 + "= "
            elif data == "ALTerm1":
                misconception = misconception + "\n"
                misconception1 = misconception1 + "\n"

            lines = str(df[data].value_counts()).splitlines()


            lines.pop()
            value = lines[0].split(" ")[0]

            if value != "none":
                misconception = misconception + value + " "

            if len(lines) > 1:
                value1 = lines[1].split(" ")[0]
                if value1 != "none":
                    misconception1 = misconception1 + value1 + " "
            else:
                if value != "none":
                    misconception1 = misconception1 + value + " " 

        print("Cluster " + str(n) + " misconception: \n" + misconception)
        print("Cluster " + str(n) + " misconception 2: \n" + misconception1)
        print("\n--------------------------------------------------\n")"""
    