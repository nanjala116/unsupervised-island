"""BasicTree.py:
    A simple implementation of a Decision Tree Classifier from scratch.
"""

import numpy as np
from collections import Counter

class TNode:
    """A node in the decision tree."""
    def __init__(self, depth):
        self.depth = depth
        self.j = None          # Feature index for the split
        self.xi = None         # Threshold value for the split
        self.left = None       # Left child node
        self.right = None      # Right child node
        self.leaf_value = None # The predicted class if the node is a leaf

def gini_impurity(y):
    """Calculates the Gini impurity for a set of labels."""
    if len(y) == 0:
        return 0
    # Count occurrences of each class
    counts = np.bincount(y)
    # Calculate probabilities
    probabilities = counts / len(y)
    # Gini impurity formula
    return 1 - np.sum(probabilities**2)

class BasicTreeClassifier:
    """
    A simple Decision Tree Classifier from scratch.
    It uses Gini impurity to find the best split.
    """
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        """Builds the decision tree from the training data."""
        self.root = self._construct_subtree(X, y, 0)

    def _construct_subtree(self, X, y, depth):
        """Recursively builds a subtree for the given data."""
        node = TNode(depth)

        # Base case: stop splitting if the node is pure, max depth is reached, or no data
        if len(np.unique(y)) == 1 or depth >= self.max_depth or len(y) == 0:
            if y.size > 0:
                node.leaf_value = Counter(y).most_common(1)[0][0]
            return node

        current_impurity = gini_impurity(y)
        best_gini = current_impurity
        best_split = None

        # Iterate through each feature and each unique value to find the best split
        for j in range(X.shape[1]):
            thresholds = np.unique(X[:, j])
            for xi in thresholds:
                left_idx = X[:, j] <= xi
                right_idx = ~left_idx
                
                y_left, y_right = y[left_idx], y[right_idx]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Calculate weighted Gini impurity of the split
                p_left = len(y_left) / len(y)
                gini = p_left * gini_impurity(y_left) + (1 - p_left) * gini_impurity(y_right)

                # If this split is better, save it
                if gini < best_gini:
                    best_gini = gini
                    best_split = {'j': j, 'xi': xi, 'left_idx': left_idx, 'right_idx': right_idx}
        
        # If a useful split was found, create child nodes
        if best_split is not None:
            node.j, node.xi = best_split['j'], best_split['xi']
            left_idx, right_idx = best_split['left_idx'], best_split['right_idx']
            
            node.left = self._construct_subtree(X[left_idx], y[left_idx], depth + 1)
            node.right = self._construct_subtree(X[right_idx], y[right_idx], depth + 1)
        else:
            # If no split improves impurity, make this a leaf node
            node.leaf_value = Counter(y).most_common(1)[0][0]
            
        return node

    def predict(self, X):
        """Makes predictions for a set of new data points."""
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        """Traverses the tree to predict a single data point."""
        if node.leaf_value is not None:
            return node.leaf_value
        
        if x[node.j] <= node.xi:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def score(self, X, y):
        """Calculates the accuracy of the model."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

