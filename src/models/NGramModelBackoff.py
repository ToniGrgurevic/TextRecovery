from collections import defaultdict
import random

from src.models.ModelInterface import ModelInterface
from collections import defaultdict, Counter


class NGramNode:
    def __init__(self):
        self.probabilities = defaultdict(float)
        self.children = defaultdict(NGramNode)


class NGramModelBackoff(ModelInterface):
    def __init__(self, n, alpha=0.1):
        self.n = n
        self.alpha = alpha
        self.root = NGramNode()
        self.vocab = set()

    def __str__(self):
        return f"n_gram_backoff_{self.n}_{self.alpha}"


    def _predict_next_char(self, context):
        node = self.root

        for char in context:
            if char in node.children and len(node.children[char].probabilities) > 0:
                node = node.children[char]
            else:
                if len(context) == 1:
                    return random.choice(list(self.vocab))
                return self._predict_next_char(context[1:])

        next_char = max(node.probabilities, key=node.probabilities.get)
        return next_char

    def predict(self, data):
        predictions = []
        for sentence in data:
            tokens = list(sentence)
            predicted_sentence = []
            for i in range(len(tokens)):
                if tokens[i] == '#':
                    context = tuple(predicted_sentence[-(self.n - 1):])
                    next_char = self._predict_next_char(context)
                    predicted_sentence.append(next_char)
                else:
                    predicted_sentence.append(tokens[i])
            predictions.append(''.join(predicted_sentence))
        return predictions

    def train(self, data, target_data):
        flat_stop = 0
        for sentence in target_data:
            tokens = list(sentence)
            self.vocab.update(tokens)
            for i in range(len(tokens)):
                node = self.root
                for j in range(i, min(i + self.n, len(tokens))):
                    char = tokens[j]
                    node.probabilities[char] += 1
                    node = node.children[char]
        self._normalize_probabilities(self.root)

    def _normalize_probabilities(self, node):
        total_count = sum(node.probabilities.values()) + self.alpha * len(node.probabilities.values())
        for char in node.probabilities:
            node.probabilities[char] = (node.probabilities[char] + self.alpha) / total_count
        for child in node.children.values():
            self._normalize_probabilities(child)
