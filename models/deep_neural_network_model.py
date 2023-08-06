import torch.nn as nn


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(40, 64)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
