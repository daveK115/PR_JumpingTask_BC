import torch.nn as nn


class SimpleModelV2(nn.Module):
    def __init__(self):
        super(SimpleModelV2, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 * 28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = self.fc_layer(x)
        return x