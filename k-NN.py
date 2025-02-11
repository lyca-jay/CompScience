import math

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.training_data = []

    def train(self, dataset):
        self.training_data = dataset

    def _euclidean_distance(self, point1, point2):
        # Ensure both points have the same number of features excluding labels for training points
        if len(point1) != len(point2) + 1:
            raise ValueError("Training points must have one more feature (the label) than test points.")
        
        # Calc Euclidean distance between two points (excluding labels)
        distance = 0
        for i in range(len(point2)):
            distance += (point1[i] - point2[i]) ** 2
        return math.sqrt(distance)

    def predict(self, test_point):
        if not self.training_data:
            raise ValueError("Training dataset is empty. Please provide training data before prediction.")

        #distances between the test point and all training points
        distances = []
        for data_point in self.training_data:
            distance = self._euclidean_distance(data_point, test_point)
            distances.append((distance, data_point[-1]))  # Store distance and label

        # Sort by distance and get the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        nearest_neighbors = distances[:self.k]

        # Count the occurrences of each class in the neighbors
        class_votes = {}
        for _, label in nearest_neighbors:
            if label not in class_votes:
                class_votes[label] = 0
            class_votes[label] += 1

        # Handle tie-breaking in votes
        max_votes = max(class_votes.values())
        top_classes = [label for label, count in class_votes.items() if count == max_votes]

        # Return the first class alphabetically in case of a tie
        return sorted(top_classes)[0]

# Example: Classifying noise level based on environment attributes
# Format: (people_count, music_volume, label)
noise_dataset = [
    (2, 3, "quiet"),
    (8, 6, "moderate"),
    (15, 9, "loud"),
    (1, 2, "quiet"),
    (10, 7, "moderate"),
    (20, 10, "loud")
]

#KNN classifier
knn = KNearestNeighbors(k=3)
knn.train(noise_dataset)

# Test with a new environment data point
test_point = (12, 8)  # 12 people, music volume level 8
result = knn.predict(test_point)
print(f'The environment with {test_point[0]} people and music volume level {test_point[1]} is classified as: {result}')
