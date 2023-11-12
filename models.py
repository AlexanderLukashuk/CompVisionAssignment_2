import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, activation_fn, num_classes, dropout_rate=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 112 * 112, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.activation_fn = activation_fn

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
                
        self.dropout = nn.Dropout(p=dropout_rate)

        # Batch Normalization layers
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(self.activation_fn(self.conv1(x)))
        x = x.view(-1, 64 * 112 * 112)

        x = self.dropout(x)

        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x
