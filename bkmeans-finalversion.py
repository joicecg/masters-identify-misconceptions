import pandas as pd
import numpy as np
from bkmeans import BKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from ExpressionTree import ExpressionTree

def extract_features_from_tree(left_tree, right_tree):
    features = {
        'total_nodes': 0,
        'operator_count': {},
        'variable_count': 0,
        'depth': 0    
    }
    
    def traverse(node, depth=1):
        if not node:
            return
        features['depth'] = max(features['depth'], depth)
        features['total_nodes'] += 1
        if node.value.isalpha():
            features['variable_count'] += 1
        else:
            if node.value not in features['operator_count']:
                features['operator_count'][node.value] = 0
            features['operator_count'][node.value] += 1
        traverse(node.left, depth + 1)
        traverse(node.right, depth + 1)
    
    traverse(left_tree)
    traverse(right_tree)

    return features

def flatten_features(features):
    flattened_features = []
    for feature in features:
        flattened_feature = {}
        for key, value in feature.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flattened_feature[f"{key}_{sub_key}"] = sub_value
            else:
                flattened_feature[key] = value
        flattened_features.append(flattened_feature)
    return flattened_features

# Load data
previous = pd.read_csv('data/tokens-previous-step.csv', encoding='utf-8')
previous.columns = ['equation', 'id']
wrong = pd.read_csv('data/tokens-wrong-step.csv', encoding='utf-8')
wrong.columns = ['equation', 'id']

# Parse equations and extract features using expression trees
def parse_and_extract_features(equations):
    features = []
    for equation in equations:
        left_tree, right_tree = ExpressionTree.parse_equation_to_tree(equation)
        features.append(extract_features_from_tree(left_tree, right_tree))
    return features

features_previous = parse_and_extract_features(previous['equation'])
features_wrong = parse_and_extract_features(wrong['equation'])

# Flatten features
features_previous_flattened = flatten_features(features_previous)
features_wrong_flattened = flatten_features(features_wrong)

# Vectorize features
from sklearn.feature_extraction import DictVectorizer

dict_vectorizer = DictVectorizer(sparse=False)
X_previous = dict_vectorizer.fit_transform(features_previous_flattened)
X_wrong = dict_vectorizer.transform(features_wrong_flattened)

# Standardize the numerical values
scaler = StandardScaler()
X_previous = scaler.fit_transform(X_previous)
X_wrong = scaler.transform(X_wrong)

# Apply BKMeans clustering
n_clusters = 100
bkmeans_previous = BKMeans(n_clusters=n_clusters)
clusters_previous = bkmeans_previous.fit_predict(X_previous)
previous['cluster'] = clusters_previous

bkmeans_wrong = BKMeans(n_clusters=n_clusters)
clusters_wrong = bkmeans_wrong.fit_predict(X_wrong)
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

with open("results/sillhouette-score.txt", 'a', encoding='utf-8') as file:
    file.write(f"BKMeans Previous - Silhouette Score: {silhouette_prev}\n")
    file.write(f"BKMeans Wrong - Silhouette Score: {silhouette_wrong}\n")
    file.write("\n\n")