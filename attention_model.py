import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_size, seq_len, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.heads = heads
        self.head_dim = seq_len // heads

        # Ensure division is clean
        assert (
                self.head_dim * heads == seq_len
        ), "Embedding size needs to be divisible by heads"

        self.reduce_dim = nn.Linear(embed_size, seq_len)

        self.queries = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(heads)])
        self.keys = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(heads)])
        self.values = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(heads)])

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, x):
        N = x.size(0)
        reduced = self.reduce_dim(x.permute(0, 2, 1)).permute(0, 2, 1)

        values_split = torch.split(reduced, self.head_dim, dim=2)
        keys_split = torch.split(reduced, self.head_dim, dim=2)
        queries_split = torch.split(reduced, self.head_dim, dim=2)

        values = []
        keys = []
        queries = []

        for i in range(self.heads):
            queries.append(self.queries[i](queries_split[i]))
            keys.append(self.keys[i](keys_split[i]))
            values.append(self.values[i](values_split[i]))

        attention_scores = torch.empty((self.heads, N, self.seq_len, self.seq_len)).to(x.device)
        for i in range(self.heads):
            attention_scores[i] = torch.bmm(queries[i], keys[i].permute(0, 2, 1))

        attention_weights = F.softmax(attention_scores, dim=-1)

        out = torch.empty((self.heads, N, self.seq_len, self.head_dim)).to(x.device)
        for i in range(self.heads):
            out[i] = torch.bmm(attention_weights[i], values[i])

        out = out.reshape(N, self.seq_len, -1)
        out = self.fc_out(out)

        return out


class ClassifierWithAttention(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=100, embed_size=500, seq_len=40, heads=8):
        super(ClassifierWithAttention, self).__init__()
        self.lstm_pre = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = SelfAttention(embed_size, seq_len, heads)
        self.lstm_post = nn.LSTM(embed_size, hidden_dim, batch_first=True)
        # self.fc1 = nn.Linear(hidden_dim * seq_len, 256)
        self.fc1 = nn.Linear(20000, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x, _ = self.lstm_pre(x)

        x = self.attention(x)
        # x, _ = self.lstm_post(x)

        x = x.reshape(x.size(0), -1)  # Changed .view() to .reshape()
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Instantiate and test
# model = ClassifierWithAttention()
# input_tensor = torch.rand(1000, 500, 40)
# output = model(input_tensor)
# print(output.shape)  # Should print torch.Size([1000, 1])
