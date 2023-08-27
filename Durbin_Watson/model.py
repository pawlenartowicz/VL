import torch
import torch.nn as nn
import torch.optim as optim
import time
# Define a generic feed-forward neural network
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, layers_list):
        super(FeedForwardNN, self).__init__()

        layers = []
        prev_dim = input_dim
        for layer_dim in layers_list:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.ReLU())
            prev_dim = layer_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# Here's a dummy dataset (modify as per your requirements)
# Let's assume X_train and y_train are input data and labels respectively
# normal distribution'
X_train = torch.randn((1000, 10))  # 1000 samples, 10 features each
y_train = torch.randn(1000, 1)  # 1000 samples


def create_residuals(X_train, layers_list = [100, 100], plot = False, no = None):
    shape = X_train.size()
    y_train = torch.randn(shape[0], 1)

    # Parameters
    input_dim = X_train.size(1)  # number of features

    model = FeedForwardNN(input_dim, layers_list)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(50):  # adjust the number of epochs as needed
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        # print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Calculate residuals
    predictions = model(X_train)
    residuals = y_train - predictions
    std = torch.std(residuals)
    residuals = residuals.detach().numpy()

    if plot:
        # Plot residuals
        import matplotlib.pyplot as plt
        plt.hist(residuals, bins=100)
        plt.show()
        # save plot
        plt.savefig(fr'D:\GitHub\VL\plots/control/residuals_{no}.png')
        plt.close()

    # calculate a standard deviation

    return residuals, std
# tensor(0.6835)
# tensor(0.7245)
