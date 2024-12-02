class HiddenMarkovModel(ModelInterface):
    def __init__(self):
        self.model = None

    def predict(self, data):
        # Implement the prediction logic for Hidden Markov model
        pass

    def train(self, data):
        # Implement the training logic for Hidden Markov model
        pass


""" ovo mi je izbacio github copilot mos vidit pa iskoristit

    !!Korist lapace smoothing!!
    
def markov_hidden():
    import json
    from hmmlearn import hmm
    import numpy as np

    def train_hmm(data, n_components=10):
        lengths = [len(sentence) for sentence in data]
        X = np.concatenate([list(map(ord, sentence)) for sentence in data]).reshape(-1, 1)
        model = hmm.MultinomialHMM(n_components=n_components, n_iter=100)
        model.fit(X, lengths)
        return model

    def predict_hmm(model, data):
        predictions = []
        for sentence in data:
            X = np.array(list(map(ord, sentence))).reshape(-1, 1)
            logprob, states = model.decode(X, algorithm="viterbi")
            predicted_sentence = ''.join(chr(state) for state in states)
            predictions.append(predicted_sentence)
        return predictions

    n_components = 10
    model = train_hmm(data, n_components)
    predicted_data = predict_hmm(model, test_data)
    metrics = accuracy(predicted_data, true_test_data)
    print(metrics)
    save(predicted_data, "markov_hidden_predictions")
    save(metrics, "markov_hidden_metrics")
"""