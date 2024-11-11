import pandas as panda
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plot
import numpy as nump
import skfuzzy as fuzzy
from collections import Counter

def load(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            columns = line.strip().split('|')
            if len(columns) >= 3:
                data.append(columns[:3])
    df = panda.DataFrame(data, columns=['Tweet_ID', 'Timestamp', 'Content'])
    return df

def create_bow_matrix(df):
    vectorizer = CountVectorizer(stop_words='english')
    bow_matrix = vectorizer.fit_transform(df['Content'])
    return bow_matrix, vectorizer

def calculate_distances(bow_matrix):
    cosine_sim_matrix = cosine_similarity(bow_matrix)
    euclidean_dist_matrix = euclidean_distances(bow_matrix)
    plot.figure(figsize=(12, 5))
    plot.subplot(1, 2, 1)
    plot.hist(cosine_sim_matrix.flatten(), bins=50, color='skyblue')
    plot.title('Cosine Similarity Distribution')
    plot.xlabel('Cosine Similarity')
    plot.ylabel('Frequency')
    plot.subplot(1, 2, 2)
    plot.hist(euclidean_dist_matrix.flatten(), bins=50, color='lightgreen')
    plot.title('Euclidean Distance Distribution')
    plot.xlabel('Euclidean Distance')
    plot.ylabel('Frequency')
    plot.tight_layout()
    plot.show()

def analyze_optimal_clusters(bow_matrix, max_clusters=10):
    inertias_kmeans = []
    silhouette_scores_kmeans = []
    silhouette_scores_agglom = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans_labels = kmeans.fit_predict(bow_matrix)
        inertias_kmeans.append(kmeans.inertia_)
        silhouette_avg_kmeans = silhouette_score(bow_matrix, kmeans_labels)
        silhouette_scores_kmeans.append(silhouette_avg_kmeans)
        agglom = AgglomerativeClustering(n_clusters=k)
        agglom_labels = agglom.fit_predict(bow_matrix.toarray())
        silhouette_avg_agglom = silhouette_score(bow_matrix, agglom_labels)
        silhouette_scores_agglom.append(silhouette_avg_agglom)

    plot.figure(figsize=(12, 5))
    plot.subplot(1, 2, 1)
    plot.plot(range(2, max_clusters + 1), inertias_kmeans, marker='o')
    plot.title('Elbow Method for K-Means')
    plot.xlabel('Number of Clusters')
    plot.ylabel('Inertia')

    plot.subplot(1, 2, 2)
    plot.plot(range(2, max_clusters + 1), silhouette_scores_kmeans, marker='o', label='K-Means', color='blue')
    plot.plot(range(2, max_clusters + 1), silhouette_scores_agglom, marker='x', label='Agglomerative', color='orange')
    plot.title('Silhouette Score Analysis')
    plot.xlabel('Number of Clusters')
    plot.ylabel('Silhouette Score')
    plot.legend()
    plot.tight_layout()
    plot.show()

def compare_silhouette_scores_all(bow_matrix, kmeans_labels, dbscan_labels, agglom_labels, fcm_labels):
    kmeans_silhouette = silhouette_score(bow_matrix, kmeans_labels)
    dbscan_silhouette = silhouette_score(bow_matrix, dbscan_labels) if len(set(dbscan_labels)) > 1 else None
    agglom_silhouette = silhouette_score(bow_matrix, agglom_labels)
    fcm_silhouette = silhouette_score(bow_matrix, fcm_labels)

    print("Silhouette Scores Comparison:")
    print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")
    if dbscan_silhouette is not None:
        print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.4f}")
    else:
        print("DBSCAN Silhouette Score: Not applicable (only one cluster or noise detected)")
    print(f"Agglomerative Clustering Silhouette Score: {agglom_silhouette:.4f}")
    print(f"Fuzzy C-Means Silhouette Score: {fcm_silhouette:.4f}")

    return {
        "K-Means": kmeans_silhouette,
        "DBSCAN": dbscan_silhouette,
        "Agglomerative": agglom_silhouette,
        "Fuzzy C-Means": fcm_silhouette
    }

def run_clustering(bow_matrix):
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans_labels = kmeans.fit_predict(bow_matrix)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(bow_matrix)
    agglom = AgglomerativeClustering(n_clusters=5)
    agglom_labels = agglom.fit_predict(bow_matrix.toarray())
    bow_matrix_transposed = bow_matrix.toarray().T
    cntr, u, _, _, _, _, _ = fuzzy.cluster.cmeans(
        bow_matrix_transposed, 5, 2, error=0.005, maxiter=1000, init=None
    )
    fcm_labels = nump.argmax(u, axis=0)
    
    return kmeans_labels, dbscan_labels, agglom_labels, fcm_labels

def visualize_clusters(bow_matrix, labels, title):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(bow_matrix.toarray())
    plot.figure(figsize=(8, 6))
    plot.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=10)
    plot.title(title)
    plot.xlabel('PCA Component 1')
    plot.ylabel('PCA Component 2')
    plot.colorbar()
    plot.show()

def print_cluster_sizes(cluster_labels):
    cluster_counts = Counter(cluster_labels)
    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count} tweets")

def print_full_sample_tweets(df, cluster_labels, cluster_number, num_samples=5):
    cluster_tweets = df['Content'][cluster_labels == cluster_number]
    print(f"\nSample tweets from Cluster {cluster_number}:")
    for i, tweet in enumerate(cluster_tweets[:num_samples], 1):
        print(f"{i}: {tweet}")

def find_most_common_words(df, cluster_labels, cluster_number, vectorizer):
    cluster_tweets = df['Content'][cluster_labels == cluster_number]
    all_words = ' '.join(cluster_tweets).lower()
    tokenized_words = all_words.split()
    stop_words = set(vectorizer.get_stop_words())
    filtered_words = [word for word in tokenized_words if word not in stop_words]
    word_counts = Counter(filtered_words)
    most_common_words = word_counts.most_common(10)
    
    return most_common_words

def calculate_entropy(labels, cluster_labels):
    entropy_sum = 0
    total_points = len(labels)
    
    for cluster in nump.unique(cluster_labels):
        cluster_points = labels[cluster_labels == cluster]
        cluster_size = len(cluster_points)
        if cluster_size == 0:
            continue
        label_counts = Counter(cluster_points)
        cluster_entropy = 0
        for count in label_counts.values():
            p = count / cluster_size
            cluster_entropy -= p * nump.log2(p)

        entropy_sum += (cluster_size / total_points) * cluster_entropy
    return entropy_sum

def calculate_purity(labels, cluster_labels):
    total_points = len(labels)
    purity_sum = 0

    for cluster in nump.unique(cluster_labels):
        cluster_points = labels[cluster_labels == cluster]
        if len(cluster_points) == 0:
            continue
        most_common_label_count = Counter(cluster_points).most_common(1)[0][1]
        purity_sum += most_common_label_count
    return purity_sum / total_points

def evaluate_clustering(labels, kmeans_labels, dbscan_labels, agglom_labels, fcm_labels):
    print("K-Means Entropy:", calculate_entropy(labels, kmeans_labels))
    print("K-Means Purity:", calculate_purity(labels, kmeans_labels))
    print("DBSCAN Entropy:", calculate_entropy(labels, dbscan_labels))
    print("DBSCAN Purity:", calculate_purity(labels, dbscan_labels))
    print("Agglomerative Entropy:", calculate_entropy(labels, agglom_labels))
    print("Agglomerative Purity:", calculate_purity(labels, agglom_labels))
    print("Fuzzy C-Means Entropy:", calculate_entropy(labels, fcm_labels))
    print("Fuzzy C-Means Purity:", calculate_purity(labels, fcm_labels))

if __name__ == "__main__":
    path = './cnnhealth.txt'
    df = load(path)
    bow_matrix, vectorizer = create_bow_matrix(df)
    true_labels = nump.random.randint(0, 5, size=len(df))
    calculate_distances(bow_matrix)
    analyze_optimal_clusters(bow_matrix)
    kmeans_labels, dbscan_labels, agglom_labels, fcm_labels = run_clustering(bow_matrix)
    evaluate_clustering(true_labels, kmeans_labels, dbscan_labels, agglom_labels, fcm_labels)
    compare_silhouette_scores_all(bow_matrix, kmeans_labels, dbscan_labels, agglom_labels, fcm_labels)
    visualize_clusters(bow_matrix, kmeans_labels, 'K-Means Clustering')
    visualize_clusters(bow_matrix, dbscan_labels, 'DBSCAN Clustering')
    visualize_clusters(bow_matrix, agglom_labels, 'Agglomerative Clustering')
    visualize_clusters(bow_matrix, fcm_labels, 'Fuzzy C-Means Clustering')
    chosen_cluster_number = 0 
    cluster_labels = kmeans_labels
    print_cluster_sizes(cluster_labels)
    print_full_sample_tweets(df, cluster_labels, chosen_cluster_number)
    common_words = find_most_common_words(df, cluster_labels, chosen_cluster_number, vectorizer)
    print("\nMost common words in Cluster", chosen_cluster_number, ":", common_words)



