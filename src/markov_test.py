from hmmlearn import hmm
import numpy as np
from sklearn.metrics import accuracy_score

from src.data.DataReader import DataReader
from hmmlearn.hmm import CategoricalHMM

data = DataReader.read("../data/corrupted_train.txt")
true_data = DataReader.read("../data/original_train.txt")

test_data = DataReader.read("../data/corrupted_test.txt")
true_test_data = DataReader.read("../data/original_test.txt")

# Function to load text data from a file
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# Function to prepare the data (states and observations)
def prepare_data(original_texts, corrupted_texts):
    states = []
    observations = []

    for original_text, corrupted_text in zip(original_texts, corrupted_texts):
        states.extend(list(original_text))  # Original characters (states)
        observations.extend(list(corrupted_text))  # Corrupted characters (observations)

    unique_states = list(set(states))  # Unique states (characters)
    unique_observations = list(set(observations))  # Unique observations (corrupted characters)

    # Mapping characters to numbers for HMM
    state_map = {s: i for i, s in enumerate(unique_states)}
    observation_map = {o: i for i, o in enumerate(unique_observations)}

    state_sequence = [state_map[s] for s in states]  # Numeric states
    observation_sequence = [observation_map[o] for o in observations]  # Numeric observations

    return state_sequence, observation_sequence, unique_states, unique_observations, state_map, observation_map

# Function to train the HMM model


def train_hmm(observation_sequence, n_states, n_observations):
    model = CategoricalHMM(n_components=n_states, n_iter=100, tol=1e-4, init_params="ste")
    model.startprob_ = np.full(n_states, 1.0 / n_states)  # Equal start probabilities
    model.transmat_ = np.full((n_states, n_states), 1.0 / n_states)  # Equal transition probabilities
    model.emissionprob_ = np.full((n_states, n_observations), 1.0 / n_observations)  # Equal emission probabilities

    # Reshape observation_sequence for HMM input format
    observation_sequence = np.array(observation_sequence).reshape(-1, 1)  # Reshape to 2D array (necessary for hmmlearn)

    # Train the model on the data
    model.fit(observation_sequence)
    return model

# Function to test the trained HMM model
def test_hmm(model, observation_sequence, unique_states, state_sequence):
    # Decode the sequence using the trained model
    log_prob, decoded_states = model.decode(observation_sequence, algorithm="viterbi")

    # Convert the decoded states back to characters
    predicted_text = "".join([unique_states[state] for state in decoded_states])

    # Calculate accuracy
    accuracy = accuracy_score(state_sequence, decoded_states)

    return predicted_text, accuracy


def main():
    # Prepare training data
    state_sequence, observation_sequence, unique_states, unique_observations, state_map, observation_map = prepare_data(
        true_data, data)

    # Train the HMM model on training data
    n_states = len(unique_states)  # Number of unique states
    n_observations = len(unique_observations)  # Number of unique observations
    model = train_hmm(observation_sequence, n_states, n_observations)

    # Test the model on the test data
    test_state_sequence, test_observation_sequence, _, _, _, _ = prepare_data(true_test_data, test_data)
    predicted_text, accuracy = test_hmm(model, test_observation_sequence, unique_states, test_state_sequence)

    # Output results
    print("Original test text (first 500 characters):", true_test_data[:500])
    print("Corrupted test text (first 500 characters):", test_data[:500])
    print("Predicted text (first 500 characters):", predicted_text[:500])
    print(f"Character-level accuracy: {accuracy * 100:.2f}%")

    # Save the results to a file
    output_path = "C:/Users/Marin/Desktop/ML/data/markov_test_output.txt"  # Path to save the output file
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(f"Original test text (first 500 characters):\n{true_test_data[:500]}\n\n")
        output_file.write(f"Corrupted test text (first 500 characters):\n{test_data[:500]}\n\n")
        output_file.write(f"Predicted text (first 500 characters):\n{predicted_text[:500]}\n\n")
        output_file.write(f"Character-level accuracy: {accuracy * 100:.2f}%\n")

    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()

