import numpy as np
from hmmlearn.hmm import CategoricalHMM
from src.models.ModelInterface import ModelInterface

class HiddenMarkovModel(ModelInterface):
    def __init__(self, n_components=58, n_iter=100):
        self.model = None
        self.n_components = n_components
        self.state_map = None
        self.observation_map = None
        self.unique_states = None
        self.unique_observations = None

    def train(self, data, target):
        # Prepare the data
        train_states, train_observations, unique_states, unique_observations, state_map, observation_map = self.prepare_data(
            target, data)
        self.state_map = state_map
        self.observation_map = observation_map
        self.unique_states = unique_states
        self.unique_observations = unique_observations

        # Set the number of components and observations
        self.n_components = len(unique_states)
        n_observations = len(unique_observations)

        # Initialize the model with the correct number of features
        self.model = CategoricalHMM(n_components=self.n_components, n_iter=500, random_state=42, init_params='')

        # Initialize counts with the correct shapes
        start_counts = np.full(self.n_components, 1 / self.n_components)
        trans_counts = np.full((self.n_components, self.n_components), 1 / self.n_components)
        emit_counts = np.full((self.n_components, n_observations), 1 / n_observations)

        # Normalize counts to obtain probabilities
        self.model.startprob_ = start_counts / start_counts.sum()
        self.model.transmat_ = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        self.model.emissionprob_ = emit_counts / emit_counts.sum(axis=1, keepdims=True)
        print(f"Emission probabilities shape: {self.model.emissionprob_.shape}")
        print(f"Expected shape: ({self.model.n_components}, {n_observations})")

        # Fit the HMM to the integer observation sequence in batches of 64 texts
        batch_size = 64
        for i in range(0, len(train_observations), batch_size):
            batch_data = train_observations[i:i + batch_size]
            batch_data = np.concatenate(batch_data).reshape(-1, 1)  # Each observation as an integer
            if np.any(np.isnan(batch_data)) or np.any(np.isinf(batch_data)):
                print(f"Invalid values found in batch starting at index {i}")
                continue
            self.model.fit(batch_data, lengths=[len(batch_data)])

    def predict(self, data):
        _, test_observations, _, _, _, _ = self.prepare_data(data, data)
        predicted_characters = []
        for test_data in test_observations:
            test_data = np.array(test_data).reshape(-1, 1)
            predicted_states = self.model.predict(test_data)
            inverse_state_map = {v: k for k, v in self.state_map.items()}
            predicted_characters.append("".join([inverse_state_map[state] for state in predicted_states]))
        return predicted_characters

    def prepare_data(self, original_texts, corrupted_texts):
        states = []
        observations = []

        for original_text, corrupted_text in zip(original_texts, corrupted_texts):
            states.extend(list(original_text))  # Original characters (states)
            observations.extend(list(corrupted_text))  # Corrupted characters (observations)

        unique_states = list(set(states))  # Unique states (characters)
        unique_observations = list(set(observations))  # Unique observations (corrupted characters)

        # Mapping characters to numbers for HMM
        self.state_map = {s: i for i, s in enumerate(unique_states)}
        self.observation_map = {o: i for i, o in enumerate(unique_observations)}

        state_sequences = [[self.state_map[s] for s in list(original_text)] for original_text in original_texts]
        observation_sequences = [[self.observation_map[o] for o in list(corrupted_text)] for corrupted_text in corrupted_texts]

        return (state_sequences, observation_sequences, unique_states, unique_observations, self.state_map,
                self.observation_map)