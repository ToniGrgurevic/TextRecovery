from abc import ABC

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from src.evaluation.evaluate import accuracy
from src.models.ModelInterface import ModelInterface


class HiddenMarkovModel2(ModelInterface):
    def predict(self, data):
        pass

    def __init__(self, n_components=58, n_iter=100, batch_size=100):
        self.n_components = n_components
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.startprob_ = np.full(n_components, 1 / n_components, dtype=np.float32)
        self.transmat_ = np.full((n_components, n_components), 1 / n_components, dtype=np.float32)
        self.emissionprob_ = None
        self.state_map = None
        self.observation_map = None
        self.unique_states = None
        self.unique_observations = None
        self.validation_accuracies = []

    def train(self, corrupted_data, true_data):
        train_data = corrupted_data[:-500]
        train_true_data = true_data[:-500]
        val_data = corrupted_data[-500:]
        val_true_data = true_data[-500:]

        train_states, train_observations, unique_states, unique_observations, state_map, observation_map = \
            self.prepare_data(train_true_data, train_data)

        self.state_map = state_map
        self.observation_map = observation_map
        self.unique_states = unique_states
        self.unique_observations = unique_observations

        self.n_components = len(unique_states)
        n_observations = len(unique_observations)
        self.emissionprob_ = np.full((self.n_components, n_observations), 1 / n_observations, dtype=np.float32)

        best_accuracy = 0

        for iteration in range(self.n_iter):

            print(f"Iteration {iteration + 1}")

            gamma, xi = self._expectation_in_batches(train_observations)

            self._maximization(train_observations, gamma, xi)

            if iteration % 5 == 0:  # Validate every 5 iterations
                val_predictions = self.predict(val_data)
                val_accuracy = accuracy(val_predictions, val_true_data)
                self.validation_accuracies.append(val_accuracy)
                print(f"Validation Accuracy: {val_accuracy['Character Accuracy']:.2f}%")

                if val_accuracy['Character Accuracy'] > best_accuracy:
                    best_accuracy = val_accuracy['Character Accuracy']
                elif best_accuracy - val_accuracy['Character Accuracy'] > 1:
                    print("Significant decrease in accuracy. Stopping training.")
                    break

        self.plot_accuracies()

    def _expectation_in_batches(self, observations):
        gamma = []
        xi_sum = np.zeros((self.n_components, self.n_components), dtype=np.float32)
        for i in range(0, len(observations), self.batch_size):
            batch_observations = observations[i:i + self.batch_size]
            batch_gamma, batch_xi_sum = self._expectation(batch_observations)
            gamma.extend(batch_gamma)
            xi_sum += batch_xi_sum  # Aggregate sufficient statistics
        return gamma, xi_sum

    def _expectation(self, observations):
        gamma = []
        xi_sum = np.zeros((self.n_components, self.n_components), dtype=np.float32)
        for obs_seq in observations:
            alpha = self._forward(obs_seq, self.startprob_, self.transmat_, self.emissionprob_)
            beta = self._backward(obs_seq, self.transmat_, self.emissionprob_)
            alpha_beta = alpha * beta
            alpha_beta_sum = alpha_beta.sum(axis=1, keepdims=True)
            alpha_beta_sum[alpha_beta_sum == 0] = 1
            gamma_seq = alpha_beta / alpha_beta_sum
            xi_sum += self._compute_xi(obs_seq, alpha, beta)
            gamma.append(gamma_seq)
        return gamma, xi_sum

    @staticmethod
    @njit
    def _forward(obs_seq, startprob, transmat, emissionprob):
        T = len(obs_seq)
        alpha = np.zeros((T, startprob.shape[0]), dtype=np.float32)
        alpha[0] = startprob * emissionprob[:, obs_seq[0]]
        for t in range(1, T):
            alpha[t] = alpha[t - 1].dot(transmat) * emissionprob[:, obs_seq[t]]
        return alpha

    @staticmethod
    @njit
    def _backward(obs_seq, transmat, emissionprob):
        T = len(obs_seq)
        beta = np.zeros((T, transmat.shape[0]), dtype=np.float32)
        beta[-1] = 1
        for t in range(T - 2, -1, -1):
            beta[t] = transmat.dot(emissionprob[:, obs_seq[t + 1]] * beta[t + 1])
        return beta

    def _compute_xi(self, obs_seq, alpha, beta):
        T = len(obs_seq)
        xi_sum = np.zeros((self.n_components, self.n_components), dtype=np.float32)
        for t in range(T - 1):
            denominator = alpha[t].dot(self.transmat_).dot(self.emissionprob_[:, obs_seq[t + 1]] * beta[t + 1])
            denominator = max(denominator, 1e-10)
            xi_t = (alpha[t][:, np.newaxis] * self.transmat_ * self.emissionprob_[:, obs_seq[t + 1]] * beta[
                t + 1]) / denominator
            xi_sum += xi_t
        return xi_sum

    def _maximization(self, observations, gamma, xi_sum):
        self.startprob_ = np.mean([g[0] for g in gamma], axis=0).astype(np.float32)
        self.transmat_ = xi_sum
        transmat_sum = self.transmat_.sum(axis=1, keepdims=True)
        transmat_sum[transmat_sum == 0] = 1
        self.transmat_ /= transmat_sum

        num = np.zeros_like(self.emissionprob_, dtype=np.float32)
        denom = np.zeros((self.n_components, 1), dtype=np.float32)
        for obs_seq, gamma_seq in zip(observations, gamma):
            obs_arr = np.array(obs_seq)
            for state in range(self.n_components):
                num[state] += np.sum(gamma_seq[:, state] * (obs_arr == state), axis=0)
            denom[:, 0] += np.sum(gamma_seq, axis=0)
        denom[denom == 0] = 1
        self.emissionprob_ = (num / denom).astype(np.float32)

    def prepare_data(self, original_texts, corrupted_texts):
        states = []
        observations = []

        for original_text, corrupted_text in zip(original_texts, corrupted_texts):
            states.extend(list(original_text))
            observations.extend(list(corrupted_text))

        unique_states = list(set(states))
        unique_observations = list(set(observations))

        self.state_map = {s: i for i, s in enumerate(unique_states)}
        self.observation_map = {o: i for i, o in enumerate(unique_observations)}

        state_sequences = [[self.state_map[s] for s in list(original_text)] for original_text in original_texts]
        observation_sequences = [[self.observation_map[o] for o in list(corrupted_text)] for corrupted_text in
                                 corrupted_texts]

        return (state_sequences, observation_sequences, unique_states, unique_observations, self.state_map,
                self.observation_map)

    def plot_accuracies(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.validation_accuracies, label='Validation Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy over Iterations')
        plt.legend()
        plt.grid(True)
        plt.savefig('validation_accuracy.png')
        plt.show()
