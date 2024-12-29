import numpy as np
from hmmlearn.hmm import CategoricalHMM
from tqdm import tqdm


def prepare_data(original_texts, corrupted_texts):
    """Prepare training data for categorical HMM."""
    states = []
    observations = []

    for original_text, corrupted_text in zip(original_texts, corrupted_texts):
        states.extend(list(original_text))
        observations.extend(list(corrupted_text))

    # Create mappings
    unique_states = sorted(list(set(states)))
    unique_observations = sorted(list(set(observations)))

    state_map = {s: i for i, s in enumerate(unique_states)}
    observation_map = {o: i for i, o in enumerate(unique_observations)}

    # Convert to numeric sequences
    state_sequence = np.array([state_map[s] for s in states])
    observation_sequence = np.array([observation_map[o] for o in observations])

    return (state_sequence, observation_sequence,
            unique_states, unique_observations,
            state_map, observation_map)


def train_hmm(train_observations, n_states, n_observations):
    """Train CategoricalHMM model."""
    # Initialize model
    model = CategoricalHMM(
        n_components=n_states,
        n_iter=100,
        random_state=42,
        init_params="ste"  # Initialize start prob, transitions, and emissions
    )

    # Set the number of possible categories
    model.n_features = 1  # Single categorical feature
    model.n_categories = [n_observations]  # Number of possible categories for the feature

    # Reshape observations for hmmlearn
    train_observations = train_observations.reshape(-1, 1)

    # Fit model with progress bar
    with tqdm(total=model.n_iter, desc="Training HMM", unit="iteration") as pbar:
        model.fit(train_observations)
        pbar.update(model.n_iter)

    return model


def load_data(file_path):
    """Load text data from file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read().splitlines()


def main():
    print("Starting text restoration process...")

    # Load data
    print("Loading training and testing data...")
    original_train_texts = load_data("../data/original_train.txt")
    corrupted_train_texts = load_data("../data/corrupted_train.txt")
    original_test_texts = load_data("../data/original_test.txt")
    corrupted_test_texts = load_data("../data/corrupted_test.txt")

    # Prepare training data
    print("Preparing training data...")
    (train_states, train_observations,
     unique_states, unique_observations,
     state_map, observation_map) = prepare_data(original_train_texts, corrupted_train_texts)

    n_states = len(unique_states)
    n_observations = len(unique_observations)
    print(f"Number of unique states: {n_states}")
    print(f"Number of unique observations: {n_observations}")

    # Train model
    print("Training HMM model...")
    model = train_hmm(train_observations, n_states, n_observations)

    # Prepare test data
    print("Preparing test data...")
    test_states, test_observations, _, _, _, _ = prepare_data(
        original_test_texts, corrupted_test_texts)

    # Predict
    print("Making predictions...")
    test_observations = test_observations.reshape(-1, 1)
    predicted_states = model.predict(test_observations)

    # Convert predictions back to text
    predicted_text = ''.join(unique_states[state] for state in predicted_states)

    # Save results
    print("Saving predictions...")
    with open('predicted_texts.txt', 'w', encoding='utf-8') as f:
        f.write(predicted_text)

    print(f"Predictions saved. Length: {len(predicted_text)}")


if __name__ == "__main__":
    main()