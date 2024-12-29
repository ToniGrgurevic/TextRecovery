from nltk.translate.bleu_score import sentence_bleu
from rapidfuzz.distance import Levenshtein


def accuracy(predicted_data, true_data):
    total_chars = 0
    correct_chars = 0
    bleu_scores = []
    levenshtein_distances = []

    for pred, true in zip(predicted_data, true_data):
        pred_tokens = list(pred)
        true_tokens = list(true)

        # Character-level accuracy
        total_chars += len(true_tokens)
        correct_chars += sum(p == t for p, t in zip(pred_tokens, true_tokens))

        # BLEU score
        bleu_score = sentence_bleu([true_tokens], pred_tokens)
        bleu_scores.append(bleu_score)

        # Levenshtein distance
        levenshtein_distance = Levenshtein.distance(pred, true)
        levenshtein_distances.append(levenshtein_distance)

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    avg_levenshtein_distance = sum(levenshtein_distances) / len(levenshtein_distances)
    char_accuracy = correct_chars / total_chars

    return {
        'Character Accuracy': char_accuracy,
        'BLEU': avg_bleu_score,
        'Levenshtein Distance': avg_levenshtein_distance
    }

