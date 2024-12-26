from src.models.ModelInterface import ModelInterface


class HiddenMarkovModel(ModelInterface):
    def __init__(self):
        self.model = None

    def predict(self, data):
        # Implement the prediction logic for Hidden Markov model
        pass

    def train(self, data, target):
        # Implement the training logic for Hidden Markov model
        pass



"""
    !!Korist lapace smoothing!!
"""
    
