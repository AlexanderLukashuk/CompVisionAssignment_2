import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, activation_fn, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 112 * 112, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.pool(self.activation_fn(self.conv1(x)))
        x = x.view(-1, 64 * 112 * 112)
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x
