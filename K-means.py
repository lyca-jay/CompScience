import random
import math

class KMeansClustering:
    def __init__(self, k=3, max_iterations=100, tolerance=1e-4):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = []

    def _initialize_centroids(self, data):
        # Randomly select k unique points as initial centroids
        self.centroids = random.sample(data, self.k)

    def _assign_clusters(self, data):
        clusters = [[] for _ in range(self.k)]
        for point in data:
            # Calculate the distance of the point to each centroid
            distances = [self._euclidean_distance(point, centroid) for centroid in self.centroids]
            # Find closest centroid and assign the point to its cluster
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(point)
        return clusters

    def _calculate_new_centroids(self, clusters):
        new_centroids = []
        for cluster in clusters:
            # Calculate the mean of each cluster to determine new centroid
            if cluster:  # Avoid division by zero
                centroid = [sum(features) / len(cluster) for features in zip(*cluster)]
                new_centroids.append(centroid)
            else:  # Retain the old centroid if cluster is empty
                # Select a random point from the data to avoid empty cluster problem
                new_centroids.append(random.choice(self.centroids))
        return new_centroids

    def _euclidean_distance(self, point1, point2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

    def _converged(self, new_centroids):
        # Check if centroids have moved less than the specified tolerance
        for i in range(self.k):
            distance = self._euclidean_distance(self.centroids[i], new_centroids[i])
            if distance > self.tolerance:
                return False
        return True

    def fit(self, data):
        # Initialize centroids
        self._initialize_centroids(data)
        
        for _ in range(self.max_iterations):
            # Assign clusters based on current centroids
            clusters = self._assign_clusters(data)
            # Calculate new centroids
            new_centroids = self._calculate_new_centroids(clusters)
            
            # Check for convergence using the updated centroids
            if self._converged(new_centroids):
                break
            self.centroids = new_centroids

        return self.centroids

# Unique example usage: Clustering students based on two exam scores
exam_scores = [
    [85, 70], [80, 65], [90, 88], [55, 45], [60, 50], [58, 48], [91, 92], [88, 85]
]

kmeans = KMeansClustering(k=2)
centroids = kmeans.fit(exam_scores)

print("Final centroids after clustering:")
for idx, centroid in enumerate(centroids):
    print(f'Centroid {idx + 1}: {centroid}')
