import numpy as np
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm
from src.data.DataReader import DataReader

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

# Function to load text data from a file
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read().splitlines()

# Function to train the HMM model in batches
def train_hmm_batched(model, train_states, train_observations, batch_size):
    num_batches = len(train_observations) // batch_size
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size

        # Get the batch data for the model
        batch_states = train_states[batch_start:batch_end]
        batch_observations = train_observations[batch_start:batch_end]

        # Reshape the observations into a 2D array of shape (n_samples, n_features)
        batch_data = np.array(batch_observations).reshape(-1, 1)

        # Fit the model with the batch data
        model.fit(batch_data)

        print(f"Batch {batch_idx+1}/{num_batches} trained")
    return model

# Function to test the model on the test data
def test_hmm_model(model, test_observations):
    test_data = np.array(test_observations).reshape(-1, 1)
    predicted_states = model.predict(test_data)
    return predicted_states

# Main function to orchestrate the process
def main():
    # Load training and testing data
    print("Loading training and testing data...")
    corrupted_train_file = DataReader.read("../data/corrupted_train.txt")
    original_train_file = DataReader.read("../data/original_train.txt")

    corrupted_test_file = DataReader.read("../data/corrupted_test.txt")
    original_test_file = DataReader.read("../data/original_test.txt")

    # Load the original and corrupted training data
    original_train_texts = DataReader.read("../data/original_train.txt")
    corrupted_train_texts = DataReader.read("../data/corrupted_train.txt")

    # Prepare the data for training
    print("Preparing training data...")
    train_states, train_observations, unique_states, unique_observations, state_map, observation_map = prepare_data(original_train_texts, corrupted_train_texts)

    # Initialize the HMM model (GaussianHMM)
    n_states = len(unique_states)  # Number of unique states
    n_features = len(unique_observations)  # Number of unique observations

    print(f"Number of unique states: {n_states}, Number of unique observations: {n_features}")
    model = GaussianHMM(n_components=n_states, n_iter=100, random_state=42, init_params="")

    # Initialize the start probabilities and transition matrix
    model.startprob_ = np.full(n_states, 1 / n_states)  # Equal start probabilities
    model.transmat_ = np.full((n_states, n_states), 1 / n_states)  # Equal transition probabilities

    # Ensure startprob_ sums to 1 and does not contain nan values
    if not np.isclose(model.startprob_.sum(), 1.0):
        model.startprob_ = np.full(n_states, 1 / n_states)

    # Batch training
    batch_size = 1000  # Adjust based on available memory and dataset size
    print("Training the HMM model in batches...")
    model = train_hmm_batched(model, train_states, train_observations, batch_size)

    # Load the testing data (corrupted version)
    print("Loading testing data...")
    original_test_texts = load_data(original_test_file)  # Original clean test data
    corrupted_test_texts = load_data(corrupted_test_file)  # Corrupted test data

    # Prepare the test data
    test_states, test_observations, _, _, _, _ = prepare_data(original_test_texts, corrupted_test_texts)

    # Test the model
    print("Testing the HMM model...")
    predicted_states = test_hmm_model(model, test_observations)

    # Convert predicted states back to characters using state_map
    predicted_texts = [''.join([unique_states[state] for state in predicted_states])]

    # Print the results
    print("Predicted Text:", predicted_texts[0])
    print(f"Predicted Text Length: {len(predicted_texts[0])}")

    # Optionally, save the results to a file
    with open('predicted_texts.txt', 'w', encoding='utf-8') as output_file:
        output_file.write(predicted_texts[0])

if __name__ == "__main__":
    print("Starting text restoration process...")
    main()