import json

from src.data.DataReader import DataReader
from src.evaluation.evaluate import accuracy
from src.models.LSTMModel import LSTMModel
from src.models.NGramModel import NGramModel
from src.validation.validate import validation_nGram

data = DataReader.read("../data/corrupted_train.txt")
true_data = DataReader.read("../data/original_train.txt")

test_data = DataReader.read("../data/corrupted_test.txt")
true_test_data = DataReader.read("../data/original_test.txt")


def save(information, file_path):
    with open("../output" + file_path, 'w', encoding="UTF-8") as f:
        f.write(str(information))


def ltsm():
    model = LSTMModel()
    model.train(data, true_data)
    predicted_data = model.predict(test_data)
    metrics = accuracy(predicted_data, true_test_data)
    print(metrics)
    save(predicted_data, "lstm_predictions")
    save(metrics, "lstm_metrics")


def n_gram():
    valid_line = len(data) - 500
    #n, alfa = validation_nGram(data[:valid_line],data[valid_line:],
     #                          true_data[:valid_line],true_data[valid_line:])
    n = 6
    alfa = 0.01
    n_grams = NGramModel(n, alfa)
    n_grams.train(data,true_data)
    predicted_data = n_grams.predict(test_data)
    metrics = accuracy(predicted_data, true_test_data)
    print(metrics)
    save(predicted_data, "n_gram_predictions")
    save(metrics, "n_gram_metrics")

def tets():
    metrics = accuracy(test_data, true_test_data)
    print(metrics)

def markov_hidden():
    pass


tets()
