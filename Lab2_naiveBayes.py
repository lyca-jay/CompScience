class NaiveBayesClassifier:
    def __init__(self):
        self.word_probs = {}
        self.class_probs = {}
        self.vocabulary = set()

    def train(self, dataset):
        class_counts = {}
        word_counts = {}
        total_documents = len(dataset)

        for text, label in dataset:
            # Initialize counts for classes
            if label not in class_counts:
                class_counts[label] = 0
                word_counts[label] = {}

            class_counts[label] += 1

            # Tokenize and count words for each class
            words = text.lower().split()
            self.vocabulary.update(words)

            for word in words:
                if word not in word_counts[label]:
                    word_counts[label][word] = 0
                word_counts[label][word] += 1

        # Compute class probabilities P(Class)
        self.class_probs = {label: count / total_documents for label, count in class_counts.items()}

        # Compute word probabilities P(Word|Class) using Laplace smoothing
        for label in word_counts:
            total_words_in_class = sum(word_counts[label].values())
            self.word_probs[label] = {}
            for word in self.vocabulary:
                word_count = word_counts[label].get(word, 0)
                self.word_probs[label][word] = (word_count + 1) / (total_words_in_class + len(self.vocabulary))

    def predict(self, text):
        words = text.lower().split()
        class_scores = {}

        for label in self.class_probs:
            # Start with the log probability of the class
            class_scores[label] = self.class_probs[label]

            # Multiply with probabilities of each word given the class
            for word in words:
                if word in self.vocabulary:
                    class_scores[label] *= self.word_probs[label].get(word, 1 / (len(self.vocabulary) + 1))

        # Return the class with the highest score
        return max(class_scores, key=class_scores.get)


# Unique example usage: Movie Review Sentiment Analysis dataset
movie_reviews = [
    ("The movie was fantastic!", "positive"),
    ("Absolutely boring and terrible", "negative"),
    ("Great storyline and superb acting", "positive"),
    ("Waste of time", "negative"),
    ("It was a wonderful experience", "positive")
]

classifier = NaiveBayesClassifier()
classifier.train(movie_reviews)

# Test with a new movie review
test_review = "It had a great plot and fantastic acting!"
result = classifier.predict(test_review)
print(f'The review "{test_review}" is classified as: {result}')
