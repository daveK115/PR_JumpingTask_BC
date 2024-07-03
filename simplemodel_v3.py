import torch.nn as nn


class SimpleModelV3(nn.Module):
    def __init__(self, dropout_rate_conv=0.25, dropout_rate_fc=0.5):
        super(SimpleModelV3, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate_conv)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 * 13 * 13, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate_fc),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = self.fc_layer(x)
        return x



