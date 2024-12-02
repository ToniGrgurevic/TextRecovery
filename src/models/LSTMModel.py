import torch
import torch.nn as nn
import torch.optim as optim
from nltk.downloader import unzip

from src.evaluation.evaluate import accuracy
from src.models.ModelInterface import ModelInterface
from src.data.DataReader import DataReader

import Levenshtein
from nltk.translate.bleu_score import sentence_bleu




class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super(Seq2SeqLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output


class LSTMModel(ModelInterface):
    def __init__(self, hidden_dim=128, embedding_dim=32, num_layers=1):
        self.optimizer = None
        self.model = None
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.char_to_index = None
        self.unique_chars = None

        self.criterion = nn.CrossEntropyLoss()

    def train(self, data, target_data, epochs=10):

        self.unique_chars = set(''.join(target_data) + '#')
        self.char_to_index = {'#': 0, '<PAD>': 1, '<UNK>': 2}
        for char in self.unique_chars:
            if char not in self.char_to_index:
                self.char_to_index[char] = len(self.char_to_index)

        self.model = Seq2SeqLSTM(len(self.char_to_index),
                                 self.embedding_dim,
                                 self.hidden_dim,
                                 len(self.char_to_index),
                                 self.num_layers)
        self.optimizer = optim.Adam(self.model.parameters())
        self.model.train()
        batch_size = len(data) // 250
        for epoch in range(epochs):
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i + batch_size]
                batch_target_data = target_data[i:i + batch_size]
                inputs, targets = self._prepare_batch(("".join(batch_data), "".join(batch_target_data)))
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()
                self.optimizer.step()
                print(f'Bach {i/40 + 1}, Loss: {loss.item()}')
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, dataa):
        self.model.eval()
        with torch.no_grad():
            inputs = self._prepare_input(dataa)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, dim=-1)

        index_to_char = {index: char for char, index in self.char_to_index.items()}
        predicted_strings = []
        for indices in predicted:
            predicted_string = ''.join(
                index_to_char[idx.item()] for idx in indices if idx.item() != self.char_to_index['<PAD>'])
            predicted_strings.append(predicted_string)

        return predicted_strings

    def _prepare_input(self, dataa):

        # Determine max sequence length
        max_length = max(len(text) for text in dataa)

        # Convert characters to indices
        input_indices = []
        for text in dataa:
            # Convert each character to its index, pad if necessary
            text_indices = [self.char_to_index.get(char, self.char_to_index['<UNK>']) for char in text]

            # Pad to max length
            padded_indices = text_indices + [self.char_to_index['<PAD>']] * (max_length - len(text_indices))

            input_indices.append(padded_indices)

        # Convert to PyTorch tensor
        return torch.tensor(input_indices, dtype=torch.long)

    def _prepare_batch(self, batch):
        input_indices = []
        target_indices = []

        input_text_indices = [self.char_to_index.get(char, self.char_to_index['<UNK>']) for char in batch[0]]
        target_text_indices = [self.char_to_index.get(char, self.char_to_index['<UNK>']) for char in batch[1]]
        inputs_tensor = torch.tensor(input_text_indices, dtype=torch.long)
        targets_tensor = torch.tensor(target_text_indices, dtype=torch.long)
        return inputs_tensor, targets_tensor

