from collections import defaultdict
import random

from src.models.ModelInterface import ModelInterface
from collections import defaultdict, Counter


class NGramModel(ModelInterface):
    def __init__(self, n, alfa=0.1):
        self.n = n
        self.alpha = alfa
        self.model = defaultdict(Counter)
        self.probabilities = defaultdict(dict)
        self.vocab = set()

    def __str__(self):
        return f"n_backoff_{self.n}_{self.alpha}"

    def _predict_next_char(self, context):
        if context in self.probabilities:
            next_char = max(self.probabilities[context], key=self.probabilities[context].get)
        else:
            next_char = random.choice(list(self.vocab))
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
        for sentence in target_data:
            tokens = list(sentence)
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                n_gram = tuple(tokens[i:i + self.n - 1])
                next_token = tokens[i + self.n - 1]
                self.model[n_gram][next_token] += 1
        for context, counter in self.model.items():
            total_count = sum(counter.values()) + self.alpha * len(self.vocab)
            self.probabilities[context] = {char: (count + self.alpha) / total_count for char, count in counter.items()}
