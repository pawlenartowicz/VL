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

class CNN_Discriminator(torch.nn.Module):
    def __init__(self):
        super(CNN_Discriminator, self).__init__()
        self.conv_disc_simple = torch.nn.Conv1d(in_channels=500, out_channels=100, kernel_size= 2,
                                                stride=1)  # 40
        self.conv_disc_short = torch.nn.Conv1d(in_channels=500, out_channels=100, kernel_size=4,
                                               stride=1)  # 38
        self.conv_disc_long = torch.nn.Conv1d(in_channels=500, out_channels=100, kernel_size=6,
                                              stride=1)  # 35
        self.linear_disc_1 = torch.nn.Linear(((39 * 100) + (37 * 100) + (35 * 100)), 300)
        self.linear_disc_2 = torch.nn.Linear(300, 1)
        self.norm = torch.nn.BatchNorm1d(100)

    def forward(self, x):
        x_simple = torch.nn.functional.relu(self.conv_disc_simple(x))
        x_short = torch.nn.functional.relu(self.conv_disc_short(x))
        x_long = torch.nn.functional.relu(self.conv_disc_long(x))
        x_simple = self.norm(x_simple)
        x_short = self.norm(x_short)
        x_long = self.norm(x_long)
        x = torch.cat((x_simple, x_short, x_long), dim=2)
        x = x.squeeze()
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.linear_disc_1(x))
        x = self.linear_disc_2(x)
        x = torch.sigmoid(x)

        return x