import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, data, target_data, char_to_index):
        self.data = data
        self.target_data = target_data
        self.char_to_index = char_to_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data[idx]
        target_text = self.target_data[idx]
        input_indices = [self.char_to_index.get(char, self.char_to_index['<UNK>']) for char in input_text]
        target_indices = [self.char_to_index.get(char, self.char_to_index['<UNK>']) for char in target_text]

        # Pad sequences to the same length
        max_length = max(len(input_indices), len(target_indices))
        input_indices += [self.char_to_index['<PAD>']] * (max_length - len(input_indices))
        target_indices += [self.char_to_index['<PAD>']] * (max_length - len(target_indices))

        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)
