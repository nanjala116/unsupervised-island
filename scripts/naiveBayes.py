"""naive_bayes.py: 
    A simple implementation of a Naive Bayes classifier for categorical features.
"""

import numpy as np

class CategoricalNB:
    """
    A simple Naive Bayes classifier for categorical features.
    It uses Laplace (add-1) smoothing to handle unseen features.
    """
    def fit(self, X, y):
        """Learns the class priors and feature likelihoods from the training data."""
        self.classes = np.unique(y)
        self.parameters = {}
        self.class_priors = {}
        n_samples = len(X)

        for c in self.classes:
            X_c = X[y == c]
            n_class_samples = len(X_c)
            
            # Calculate class prior probability: P(class)
            self.class_priors[c] = n_class_samples / n_samples
            self.parameters[c] = {}
            
            # Calculate feature likelihoods: P(feature | class)
            for col_idx in range(X.shape[1]):
                feature_values, counts = np.unique(X_c[:, col_idx], return_counts=True)
                n_unique_feature_values = len(np.unique(X[:, col_idx]))
                
                # Store probabilities with Laplace smoothing
                self.parameters[c][col_idx] = {val: (cnt + 1) / (n_class_samples + n_unique_feature_values) 
                                               for val, cnt in zip(feature_values, counts)}

    def _predict_single(self, x):
        """Predicts the class for a single data point using log probabilities."""
        posteriors = []
        for c in self.classes:
            # Start with log of the prior probability
            prior = np.log(self.class_priors[c])
            
            likelihood = 0
            for col_idx, feature_val in enumerate(x):
                n_unique_feature_values = len(self.parameters[c][col_idx])
                
                # Get the probability of the feature value, applying smoothing for unseen values
                prob = self.parameters[c][col_idx].get(feature_val, 1 / (len(x) + n_unique_feature_values))
                likelihood += np.log(prob)
            
            # Posterior = Prior + Likelihood (in log space)
            posterior = prior + likelihood
            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """Makes predictions for a set of new data points."""
        return np.array([self._predict_single(x) for x in X])

    def score(self, X, y):
        """Calculates the accuracy of the model."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

