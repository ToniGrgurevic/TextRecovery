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
import torch.nn.functional as F


class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, device=None):
        super(Seq2SeqLSTM, self).__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.to(self.device)
        # Convert indices to one-hot vectors
        x_onehot = F.one_hot(x, num_classes=self.input_dim).float()
        lstm_out, _ = self.lstm(x_onehot)
        output = self.fc(lstm_out)
        return output


class LSTMModelHot(ModelInterface):

    def __init__(self, hidden_dim=64, num_layers=3, learning_rate=0.001, label_smoothing=0.0):
        self.optimizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.char_to_index = None
        self.unique_chars = None
        self.label_smoothing = label_smoothing
        self.embedding_dim = 0

        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def __str__(self):
        return (f"hid{self.hidden_dim}_emb{self.embedding_dim}_lay{self.num_layers}_lr{self.learning_rate}"
                f"_ls{self.label_smoothing}")

    def train(self, data, target_data, epochs=10):
        valid_line = len(data) - 500

        self.unique_chars = set(''.join(target_data) + '#')
        self.char_to_index = {'#': 0, '<PAD>': 1, '<UNK>': 2}
        for char in self.unique_chars:
            if char not in self.char_to_index:
                self.char_to_index[char] = len(self.char_to_index)
        vocab_size = len(self.char_to_index)

        self.model = Seq2SeqLSTM(vocab_size,
                                 self.hidden_dim,
                                 vocab_size,
                                 self.num_layers,
                                 device=self.device).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        batch_size = 64
        dataset = TextDataset(data[:valid_line], target_data[:valid_line], self.char_to_index)
        dataset_testing = TextDataset(data[valid_line:], target_data[valid_line:], self.char_to_index)

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
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()
                self.optimizer.step()
                # print(f'Loss: {loss.item()}')
                loss_epoch += loss.item()

            # Compute the loss and accuracy on dataset_testing
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in dataloader_testing:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                    test_loss += loss.item()

                    _, predicted = torch.max(outputs, dim=-1)
                    total += targets.size(0) * targets.size(1)
                    correct += (predicted == targets).sum().item()

            test_loss /= len(dataloader_testing)
            accuracy = correct / total
            print(
                f'Epoch {epoch + 1}/{epochs}, Loss: {loss_epoch / len(dataloader)}, Test Loss: {test_loss}, Accuracy: {accuracy}')
            train_losses.append(loss_epoch / len(dataloader))
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            if test_loss > best_loss * 1.1:
                print("Test loss increased, stopping training.")
                break
            if test_loss < best_loss:
                best_loss = test_loss

        # Save losses and accuracies to files
        with open('../logs/TL' + self.__str__() + '.json', 'w') as f:
            json.dump(train_losses, f)
        with open('../logs/VL' + self.__str__() + '.json', 'w') as f:
            json.dump(test_losses, f)
        with open('../logs//A' + self.__str__() + '.json', 'w') as f:
            json.dump(accuracies, f)
        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'../figs/LSTM_{self.num_layers}_layers_{self.hidden_dim}_hidden_dim_losses.png')
        plt.show()

        # Plot accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(accuracies, label='Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'../figs/LSTM_{self.num_layers}_layers_{self.hidden_dim}_hidden_dim_accuracy.png')
        plt.show()

    def predict(self, test_data):
        self.model.eval()
        predictions = []
        index_to_char = {idx: char for char, idx in self.char_to_index.items()}

        with torch.no_grad():
            for seq in test_data:
                input_tensor = torch.tensor([self.char_to_index.get(char, self.char_to_index['<UNK>'])
                                             for char in seq],
                                            dtype=torch.long).unsqueeze(0).to(self.device)

                output = self.model(input_tensor)
                predicted_indices = output.argmax(dim=-1).squeeze(0)

                predicted_chars = ''.join(index_to_char[idx.item()]
                                          for idx in predicted_indices)
                predictions.append(predicted_chars)

        return predictions

    def _prepare_input(self, dataa):
        # Determine max sequence length
        max_length = max(len(text) for text in dataa)

        # Convert characters to indices
        input_indices = []
        for text in dataa:
            text_indices = [self.char_to_index.get(char, self.char_to_index['<UNK>']) for char in text]

            padded_indices = text_indices + [self.char_to_index['<PAD>']] * (max_length - len(text_indices))

            input_indices.append(padded_indices)

        return torch.tensor(input_indices, dtype=torch.long)

    def _prepare_batch(self, batch):
        input_indices = []
        target_indices = []

        input_text_indices = [self.char_to_index.get(char, self.char_to_index['<UNK>']) for char in batch[0]]
        target_text_indices = [self.char_to_index.get(char, self.char_to_index['<UNK>']) for char in batch[1]]
        inputs_tensor = torch.tensor(input_text_indices, dtype=torch.long)
        targets_tensor = torch.tensor(target_text_indices, dtype=torch.long)
        return inputs_tensor, targets_tensor
