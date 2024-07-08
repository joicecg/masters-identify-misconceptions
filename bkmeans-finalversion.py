import pandas as pd
import numpy as np
from bkmeans import BKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load data
previous = pd.read_csv('data/tokens-previous-step.csv', encoding='utf-8')
previous.columns = ['equation', 'id']
wrong = pd.read_csv('data/tokens-wrong-step.csv', encoding='utf-8')
wrong.columns = ['equation', 'id']

# Extract unique characters from both datasets, ignoring spaces
unique_chars = set("".join(previous['equation'].tolist() + wrong['equation'].tolist()))
unique_chars.discard(' ')

# Create a dictionary mapping each unique character to an integer
char_to_int = {char: idx + 1 for idx, char in enumerate(unique_chars)}  # start indexing from 1 to avoid 0
int_to_char = {idx: char for char, idx in char_to_int.items()}

# Tokenize the equations into numerical values using the dictionary, ignoring spaces
def tokenize_equation(equation, char_to_int):
    return [char_to_int[char] for char in equation if char != ' ']

previous['tokenized'] = previous['equation'].apply(lambda x: tokenize_equation(x, char_to_int))
wrong['tokenized'] = wrong['equation'].apply(lambda x: tokenize_equation(x, char_to_int))

print(previous)

# Since KMeans expects fixed-size input, we need to pad or truncate sequences
def pad_sequences(sequences, maxlen, padding='post'):
    padded_sequences = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            if padding == 'post':
                padded_sequences[i, :] = seq[:maxlen]
            else:
                padded_sequences[i, :] = seq[-maxlen:]
        else:
            if padding == 'post':
                padded_sequences[i, :len(seq)] = seq
            else:
                padded_sequences[i, -len(seq):] = seq
    return padded_sequences

# Pad sequences to the same length
max_len = max(previous['tokenized'].apply(len).max(), wrong['tokenized'].apply(len).max())
previous['tokenized_padded'] = list(pad_sequences(previous['tokenized'].tolist(), maxlen=max_len, padding='post'))
wrong['tokenized_padded'] = list(pad_sequences(wrong['tokenized'].tolist(), maxlen=max_len, padding='post'))

# Convert the padded sequences to numpy arrays
X_previous = np.array(previous['tokenized_padded'].tolist())
X_wrong = np.array(wrong['tokenized_padded'].tolist())

# Standardize the numerical values
scaler = StandardScaler()
X_previous = scaler.fit_transform(X_previous)
X_wrong = scaler.transform(X_wrong)

print(X_previous)
print(X_wrong)

# Apply K-Means clustering
n_clusters = 200
kmeans_previous = BKMeans(n_clusters=n_clusters)
clusters_previous = kmeans_previous.fit_predict(X_previous)
previous['cluster'] = clusters_previous

kmeans_wrong = BKMeans(n_clusters=n_clusters)
clusters_wrong = kmeans_wrong.fit_predict(X_wrong)
wrong['cluster'] = clusters_wrong

# Combine datasets based on id and wrong cluster
combined = pd.merge(previous, wrong, on='id', suffixes=('_previous', '_wrong'))

# Group by wrong clusters and save to file
def save_clusters_by_wrong_cluster(combined, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for cluster_num in combined['cluster_wrong'].unique():
            cluster_data = combined[combined['cluster_wrong'] == cluster_num]
            file.write(f"Cluster {cluster_num}:\n")
            file.write("previous,wrong,id,cluster\n")
            for _, row in cluster_data.iterrows():
                file.write(f"{row['equation_previous']},{row['equation_wrong']},{row['id']},{row['cluster_wrong']}\n")
            file.write("\n\n")

save_clusters_by_wrong_cluster(combined, 'results/bkmeans-clusters.csv')

# Evaluate clustering performance
def evaluate_clustering(X, clusters):
    silhouette_avg = silhouette_score(X, clusters)
    return silhouette_avg

silhouette_prev = evaluate_clustering(X_previous, clusters_previous)
silhouette_wrong = evaluate_clustering(X_wrong, clusters_wrong)

print(f"BKMeans Previous - Silhouette Score: {silhouette_prev}")
print(f"BKMeans Wrong - Silhouette Score: {silhouette_wrong}")
