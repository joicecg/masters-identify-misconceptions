import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from kmodes.kmodes import KModes
from ExpressionTree import ExpressionTree
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score


def parse_and_extract_features(equations):
    features = []
    for equation in equations:
        left_tree, right_tree = ExpressionTree.parse_equation_to_tree(equation)
        features.append(extract_features_from_tree(left_tree, right_tree))
    return features

def extract_features_from_tree(left_tree, right_tree):
    features = {
        'total_nodes': 0,
        'operator_count': {},
        'numeric_count': 0,
        'variable_count': 0,
        'depth': 0
    }
    
    def traverse(node, depth=1):
        if not node:
            return
        features['depth'] = max(features['depth'], depth)
        features['total_nodes'] += 1
        if node.value.isdigit():
            features['numeric_count'] += 1
        elif node.value.isalpha():
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

# Parse equations and extract features
features_previous = parse_and_extract_features(previous['equation'])
features_wrong = parse_and_extract_features(wrong['equation'])

# Flatten features
features_previous_flattened = flatten_features(features_previous)
features_wrong_flattened = flatten_features(features_wrong)

# Vectorize features
dict_vectorizer = DictVectorizer(sparse=False)
X_previous = dict_vectorizer.fit_transform(features_previous_flattened)
X_wrong = dict_vectorizer.transform(features_wrong_flattened)

# Apply K-Modes clustering
n_clusters = 100
km_previous = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=1)
clusters_previous = km_previous.fit_predict(X_previous)
previous['cluster'] = clusters_previous

km_wrong = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=1)
clusters_wrong = km_wrong.fit_predict(X_wrong)
wrong['cluster'] = clusters_wrong

# Combine datasets based on id and wrong cluster
combined = pd.merge(previous, wrong, on='id', suffixes=('_previous', '_wrong'))

# Group by wrong clusters and save to file
def save_clusters_by_wrong_cluster(combined, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for cluster_num in combined['cluster_wrong'].unique():
            cluster_data = combined[combined['cluster_wrong'] == cluster_num]
            file.write(f"Cluster {cluster_num}:\n")
            file.write("previous equation,wrong equation,id,cluster\n")
            for _, row in cluster_data.iterrows():
                file.write(f"{row['equation_previous']},{row['equation_wrong']},{row['id']},{row['cluster_wrong']}\n")
            file.write("\n\n")

save_clusters_by_wrong_cluster(combined, 'results/kmodes-clusters.csv')

# Evaluate clustering performance
def evaluate_clustering(X, clusters, method_name):
    silhouette_avg = silhouette_score(X, clusters)
    davies_bouldin_avg = davies_bouldin_score(X, clusters)
    return silhouette_avg, davies_bouldin_avg

silhouette_prev, db_kmeans_prev = evaluate_clustering(X_previous, clusters_previous, "KMeans Previous")
silhouette_wrong, db_kmeans_wrong = evaluate_clustering(X_wrong, clusters_wrong, "KMeans Wrong")

print(f"KModes Previous - Silhouette Score: {silhouette_prev}, Davies-Bouldin Index: {db_kmeans_prev}")
print(f"KModes Wrong - Silhouette Score: {silhouette_wrong}, Davies-Bouldin Index: {db_kmeans_wrong}")
