import torch
import torch.nn as nn
import torch.optim as optim


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, layers_list):
        super(DNN, self).__init__()

        layers = []
        prev_dim = input_dim
        for layer_dim in layers_list:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.ReLU())
            prev_dim = layer_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def create_dataset(residuals, sliding):
    X = []
    y = []
    input_window, output_window = sliding

    for i in range(len(residuals) - input_window - output_window + 1):
        X.append(residuals_tensor[i:i + input_window])
        y.append(residuals_tensor[i + input_window:i + input_window + output_window])

    return torch.stack(X), torch.stack(y)


# Assuming residuals is a list of residuals from your regression
residuals = [...]  # Fill this in
residuals_tensor = torch.tensor(residuals, dtype=torch.float32)

# Specify sliding window
sliding = [2, 2]
X, y = create_dataset(residuals, sliding)

# Model parameters
input_dim = sliding[0]
output_dim = sliding[1]
layers_list = [100, 200]  # modify as needed
model = DNN(input_dim, output_dim, layers_list)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):  # adjust the number of epochs as needed
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# If the loss is low, then there might be autocorrelation in the residuals.
