import numpy as np
from torch import nn
import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, number_of_responses):
        self.training = False
        if 'label' in df.columns:
            self.labels = [df['label'][i] for i in range(0, len(df), number_of_responses)]
            self.training = True

        self.matrices = [np.array(df.iloc[i:i+number_of_responses, :-2]) for i in range(0, len(df), number_of_responses)]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.matrices)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_matrices(self, idx):
        # Fetch a batch of inputs
        return self.matrices[idx]

    def __getitem__(self, idx):

        batch_matrices = self.get_batch_matrices(idx)
        if self.training:
            batch_labels = self.get_batch_labels(idx)
            return batch_matrices, batch_labels

        return batch_matrices

class Simple_Net(nn.Module):

    def __init__(self, dropout, input_size, layer_sizes, num_classes):

        super(Simple_Net, self).__init__()

        self.layer_sizes = layer_sizes

        self.start_layer = nn.Linear(input_size, layer_sizes[0])

        for idx, layer_dim in enumerate(layer_sizes):
            if idx == 0:
                continue
            setattr(self, f'linear_{idx}', nn.Linear(layer_sizes[idx-1], layer_sizes[idx]))

        self.final = nn.Linear(layer_sizes[-1], num_classes)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x = self.flatten(x)

        x = self.start_layer(x)

        for idx, layer_dim in enumerate(self.layer_sizes):
            if idx == 0:
                continue
            x = self.dropout(x)
            x = getattr(self, f'linear_{idx}')(x)
            x = self.relu(x)

        x = self.final(x)
        x = self.sigmoid(x)

        return x