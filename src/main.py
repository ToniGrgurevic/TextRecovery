import json

from safetensors import torch
import torch
from src.data.DataReader import DataReader
from src.evaluation.evaluate import accuracy
from src.models.LSTMModel import LSTMModel
from src.models.LSTMModelHot import LSTMModelHot
from src.models.NGramModel import NGramModel
from src.models.NGramModelBackoff import NGramModelBackoff
from src.validation.validate import validation_nGram, hyperparameter_search

data = DataReader.read("../data/corrupted_train.txt")
true_data = DataReader.read("../data/original_train.txt")

test_data = DataReader.read("../data/corrupted_test.txt")
true_test_data = DataReader.read("../data/original_test.txt")


def save(information, file_path):
    with open("../logs/" + file_path, 'w', encoding="UTF-8") as f:
        f.write(str(information))


def load_model(file_path):
    # Initialize the model with the same parameters used during training
    model = LSTMModel(hidden_dim=64, embedding_dim=64, num_layers=3, learning_rate=0.001, label_smoothing=0.1)

    # Load the state dictionary
    state_dict = torch.load(file_path)
    model.model.load_state_dict(state_dict)

    return model


def ltsm():
    model = LSTMModel()
    model.train(data, true_data)

    predicted_data = model.predict(test_data)
    metrics = accuracy(predicted_data, true_test_data)
    print(metrics)
    save(predicted_data, model.__str__() + "_predictions")
    save(metrics, model.__str__() + "_metrics")

    file_path_dict = {"path": model.__str__()}
    # Save the model state dictionary with the correct argument type
    torch.save(model.model.state_dict(), file_path_dict)

def ltsm_hot():
    model = LSTMModelHot()
    model.train(data, true_data)

    predicted_data = model.predict(test_data)
    metrics = accuracy(predicted_data, true_test_data)
    print(metrics)
    save(predicted_data, model.__str__() + "_predictions")
    save(metrics, model.__str__() + "_metrics")

    file_path_dict = {"path": model.__str__()}
    # Save the model state dictionary with the correct argument type
    torch.save(model.model.state_dict(), file_path_dict)


def ltsm3():
    model = LSTMModel()
    file_path = "../saved_models/hid64_emb64_lay3_lr0.001_ls0.1.pth"
    loaded_model = load_model(file_path)
    #model.train(data, true_data)

    predicted_data = loaded_model.predict(true_data[-500:])
    print(predicted_data)
    metrics = accuracy(true_data[-500:], true_test_data[-500:])
    save(predicted_data, loaded_model.__str__() + "_Predictions")
    print(metrics)



def ltsm2():
    best_model, best_metrics, best_params = hyperparameter_search(data, true_data)
    print("Best model trained and saved.")
    predicted_data = best_model.predict(test_data)
    metrics = accuracy(predicted_data, true_test_data)
    save(predicted_data, best_model.__str__() + "_Predictions")
    save(metrics, best_model.__str__() + "_Metrics")


def n_gram():
    valid_line = len(data) - 500

    # pronalazenje najboljeg n-grama
    n, alfa = validation_nGram(data[:valid_line], data[valid_line:],
                               true_data[:valid_line], true_data[valid_line:])
    n = 6
    alfa = 0
    n_grams = NGramModel(n, alfa)
    n_grams.train(data, true_data)
    predicted_data = n_grams.predict(test_data)
    metrics = accuracy(predicted_data, true_test_data)
    print(metrics)
    save(predicted_data, n_grams.__str__() + "_pred")
    save(metrics, n_grams.__str__() + "_metrics")


def backoff_n_gram():
    valid_line = len(data) - 500

    # pronalazenje najboljeg n-grama
    #n, alfa = validation_nGram(data[:valid_line], data[valid_line:],
     #                          true_data[:valid_line], true_data[valid_line:])
    alfa = 0
    n_grams = NGramModelBackoff(6, 0)
    n_grams.train(data, true_data)
    predicted_data = n_grams.predict(test_data)
    metrics = accuracy(predicted_data, true_test_data)
    print(metrics)
    save(predicted_data, n_grams.__str__() + "_pred")
    save(metrics, n_grams.__str__() + "_metrics")


def test():
    valid_line = len(data) - 500
    metrics = accuracy(test_data, true_test_data)
    print(metrics)


def markov_hidden():
    pass


backoff_n_gram()
