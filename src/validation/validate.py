import torch
from matplotlib import pyplot as plt
from rapidfuzz.distance import Levenshtein

from src.evaluation.evaluate import accuracy
from src.models.LSTMModel import LSTMModel
from src.models.NGramModel import NGramModel

def save(information, file_path):
    with open("../logs/" + file_path, 'w', encoding="UTF-8") as f:
        f.write(str(information))


def hyperparameter_search(data, true_data):
    hyperparameter_grid = {
        "hidden_dim": [32, 64, 128],
        "embedding_dim": [16, 32, 64],
        "num_layers": [1, 2, 3],
        "learning_rate": [0.001],
        "label_smoothing": [0.0, 0.1]
    }

    best_model = None
    best_metrics = None
    best_params = None

    for hidden_dim in hyperparameter_grid["hidden_dim"]:
        for embedding_dim in hyperparameter_grid["embedding_dim"]:
            for num_layers in hyperparameter_grid["num_layers"]:
                for learning_rate in hyperparameter_grid["learning_rate"]:
                    for label_smoothing in hyperparameter_grid["label_smoothing"]:
                        print(f"Training with params: hidden_dim={hidden_dim}, "
                              f"embedding_dim={embedding_dim}, num_layers={num_layers}, "
                              f"learning_rate={learning_rate}, label_smoothing={label_smoothing}")

                        # Create and train model
                        model = LSTMModel(hidden_dim=hidden_dim,
                                          embedding_dim=embedding_dim,
                                          num_layers=num_layers,
                                          # learning_rate=learning_rate,
                                          label_smoothing=label_smoothing)
                        model.train(data, true_data)

                        valid_line = 500
                        predicted_data = model.predict(data[:valid_line])
                        metrics = accuracy(predicted_data, true_data[:valid_line])
                        print(f"Metrics: {metrics}")
                        save(metrics, model.__str__() + "_Metrics")

                        # Save best model
                        if not best_metrics or metrics["Character Accuracy"] > best_metrics["Character Accuracy"]:
                            best_metrics = metrics
                            best_model = model
                            best_params = {
                                "hidden_dim": hidden_dim,
                                "embedding_dim": embedding_dim,
                                "num_layers": num_layers,
                                "learning_rate": learning_rate,
                                "label_smoothing": label_smoothing
                            }
                            print(best_model.__str__() + " is now best model")
                        torch.save(model.model.state_dict(), "../saved_models/" + model.__str__() + ".pth")

    # Save the best model
    torch.save(best_model.model.state_dict(), "best_lstm_model.pth")
    print("#" * 20)
    print(f"Best parameters: {best_params}")
    print(f"Best metrics: {best_metrics}")
    return best_model, best_metrics, best_params


def loss(predicted_data, true_data):
    levenshtein_distances = []

    for pred, true in zip(predicted_data, true_data):
        levenshtein_distance = Levenshtein.distance(pred, true)
        levenshtein_distances.append(levenshtein_distance)

    avg_levenshtein_distance = sum(levenshtein_distances) / len(levenshtein_distances)
    return avg_levenshtein_distance


def validation_nGram(train_data, validate_data, train_target, validate_target):
    """
    :return: n, alpha
    n range: 1-10
    alpha : [0.01, 0.1, 0.5, 1]
    function tries every parameter combination and returns model with best score in validation dataset
        score calculated with score function.
    """
    best_loss = None
    best_n = None
    losses = []
    for n in range(1, 9):
        for alpha in [0]:
            model = NGramModel(n, alpha)
            model.train(train_data, train_target)
            predicted_data = model.predict(validate_data)
            current_loss = loss(predicted_data, validate_target)
            losses.append((n, current_loss))
            print(f"Levenshtein Distance for {n}-gram : {current_loss}")
            if best_loss is None or current_loss < best_loss:
                best_loss = current_loss
                best_n = n
                best_alpha = alpha
    print(f"Best model in n-gram  is n = {best_n} -> Levenshtein Distance : {best_loss}")
    plt.figure(figsize=(10, 6))
    losses.sort()
    plt.plot([n for n, _ in losses], [loss for _, loss in losses], label='Levenshtein Distance')
    plt.xlabel('n')
    plt.ylabel('Levenshtein Distance')
    plt.title('Levenshtein Distance for different n values')
    plt.legend()
    plt.savefig('n_gram_loss.png')
    plt.show()

    return best_n, best_alpha


