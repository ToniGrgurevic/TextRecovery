from rapidfuzz.distance import Levenshtein

from src.models.NGramModel import NGramModel


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
    n range: 1-6
    alpha : [0.01, 0.1, 0.5, 1]
    function tries every parameter combination and returns model with best score in validation dataset
        score calculated with score function.
    """
    best_loss = None
    best_n = None
    best_alpha = None

    for n in range(1, 7):
        for alpha in [0.01, 0.1, 0.5, 1]:
            model = NGramModel(n, alpha)
            model.train(train_data, train_target)
            predicted_data = model.predict(validate_data)
            current_loss = loss(predicted_data, validate_target)
            print(f"loss for ({n},{alpha}) = {current_loss}")
            if best_loss is None or current_loss < best_loss:
                best_loss = current_loss
                best_n = n
                best_alpha = alpha
    print(f"Best model in n-gram family is (n, alpha) = ({best_n},{best_alpha})")
    return best_n, best_alpha
