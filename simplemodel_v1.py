import torch.nn as nn


class SimpleModelV1(nn.Module):
    def __init__(self):
        super(SimpleModelV1, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 * 29 * 29, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = self.fc_layer(x)
        return x
