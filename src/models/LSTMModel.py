import json

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from nltk.downloader import unzip
from torch.utils.data import DataLoader

from src.data.text_dataset import TextDataset
from src.evaluation.evaluate import accuracy
from src.models.ModelInterface import ModelInterface
from src.data.DataReader import DataReader

import Levenshtein
from nltk.translate.bleu_score import sentence_bleu


class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers=1, device=None):
        super(Seq2SeqLSTM, self).__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.to(self.device)
        # print(x)
        embedded = self.embedding(x)
        # print(embedded)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output


class LSTMModel(ModelInterface):
    def __init__(self, hidden_dim=64, embedding_dim=24, num_layers=1, learning_rate=0.001, label_smoothing=0.0):
        self.optimizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.label_smoothing = label_smoothing
        self.char_to_index = None
        self.unique_chars = None
        self.label_smoothing = label_smoothing

        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def __str__(self):
        return (f"hid{self.hidden_dim}_emb{self.embedding_dim}_lay{self.num_layers}_lr{self.learning_rate}"
                f"_ls{self.label_smoothing}")

    def train(self, data, target_data, epochs=10):
        valid_line = 500

        self.unique_chars = set(''.join(target_data) + '#')
        self.char_to_index = {'#': 0, '<PAD>': 1, '<UNK>': 2}
        for char in self.unique_chars:
            if char not in self.char_to_index:
                self.char_to_index[char] = len(self.char_to_index)

        self.model = Seq2SeqLSTM(len(self.char_to_index),
                                 self.embedding_dim,
                                 self.hidden_dim,
                                 len(self.char_to_index),
                                 self.num_layers,
                                 device=self.device).to(self.device)

        # self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        batch_size = 50
        dataset = TextDataset(data[valid_line:], target_data[valid_line:], self.char_to_index)
        dataset_testing = TextDataset(data[:valid_line], target_data[:valid_line], self.char_to_index)

        def collate_fn(batch):
            inputs, targets = zip(*batch)
            max_length = max(max(len(input_seq), len(target_seq)) for input_seq, target_seq in batch)
            padded_inputs = [torch.cat([input_seq,
                                        torch.tensor([self.char_to_index['<PAD>']] * (max_length - len(input_seq)),
                                                     dtype=torch.long)]) for input_seq in inputs]
            padded_targets = [torch.cat([target_seq,
                                         torch.tensor([self.char_to_index['<PAD>']] * (max_length - len(target_seq)),
                                                      dtype=torch.long)]) for target_seq in targets]
            return torch.stack(padded_inputs), torch.stack(padded_targets)

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        dataloader_testing = DataLoader(dataset_testing, batch_size=batch_size, collate_fn=collate_fn)

        train_losses = []
        test_losses = []
        accuracies = []
        best_loss = float('inf')
        self.model.train()
        for epoch in range(epochs):
            loss_epoch = 0
            for inputs, targets in dataloader:
                # Traning
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()
                self.optimizer.step()
                # print(f'Loss: {loss.item()}')
                loss_epoch += loss.item()

            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in dataloader_testing:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)

                    # Create mask for non-padding tokens
                    pad_mask = targets != self.char_to_index['<PAD>']

                    # Calculate loss only on non-padding tokens
                    outputs_masked = outputs[pad_mask]
                    targets_masked = targets[pad_mask]
                    loss = self.criterion(outputs_masked.view(-1, outputs.size(-1)),
                                          targets_masked.view(-1))
                    test_loss += loss.item()

                    # Calculate accuracy only on non-padding tokens
                    _, predicted = torch.max(outputs, dim=-1)
                    total += pad_mask.sum().item()
                    correct += ((predicted == targets) & pad_mask).sum().item()

            test_loss /= len(dataloader_testing)
            accuracy_valid = correct / total
            print(
                f'Epoch {epoch + 1}/{epochs}, Loss: {loss_epoch / len(dataloader)}, Test Loss: {test_loss},'
                f' Accuracy: {accuracy_valid}')
            train_losses.append(loss_epoch / len(dataloader))
            test_losses.append(test_loss)
            accuracies.append(accuracy_valid)

            if test_loss > best_loss * 1.1:
                print("Test loss increased, stopping training.")
                break
            if test_loss < best_loss:
                best_loss = test_loss

        with open('../logs/TL' + self.__str__() + '.json', 'w') as f:
            json.dump(train_losses, f)
        with open('../logs/VL' + self.__str__() + '.json', 'w') as f:
            json.dump(test_losses, f)
        with open('../logs//A' + self.__str__() + '.json', 'w') as f:
            json.dump(accuracies, f)
            """
        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'../figs/LSTM_' + self.__str__() + '_loss.png')
        plt.show()

        # Plot accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(accuracies, label='Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'../figs/LSTM_' + self.__str__() + '_acc.png')
        plt.show()
        """

    def predict(self, test_data):
        """
        Predicts the next characters for the given test data.

        Args:
            test_data (list of str): List of input sequences as strings.

        Returns:
            list of str: Predicted sequences as strings.
        """
        self.model.eval()
        predictions = []
        index_to_char = {idx: char for char, idx in self.char_to_index.items()}  # Reverse mapping

        with torch.no_grad():
            for seq in test_data:
                # Convert string input to tensor
                input_tensor = torch.tensor([self.char_to_index.get(char, self.char_to_index['<UNK>']) for char in seq],
                                            dtype=torch.long).unsqueeze(0).to(self.device)  # Shape: [1, seq_len]

                # Pass tensor to the model
                output = self.model(input_tensor)  # Shape: [1, seq_len, output_dim]
                predicted_indices = output.argmax(dim=-1).squeeze(0)  # Shape: [seq_len]

                # Convert tensor back to string
                predicted_chars = ''.join(index_to_char[idx.item()] for idx in predicted_indices)
                predictions.append(predicted_chars)

        return predictions

    def _prepare_input(self, dataa):
        max_length = max(len(text) for text in dataa)

        #  characters -> indices
        input_indices = []
        for text in dataa:
            text_indices = [self.char_to_index.get(char, self.char_to_index['<UNK>']) for char in text]

            padded_indices = text_indices + [self.char_to_index['<PAD>']] * (max_length - len(text_indices))

            input_indices.append(padded_indices)

        return torch.tensor(input_indices, dtype=torch.long)


