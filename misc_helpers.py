import torch
import numpy as np


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """Calculate convolution output dimensions"""
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def get_device():
    """Get device to use for torch"""
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(dev)

    return device


def setup_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_positions(training_positions, validation_positions):
    """Print out training and validation positions in environment space"""
    for floor_height in range(10, 21):
        grid = ''
        for obs_pos in range(20, 46):
            if (obs_pos, floor_height) in training_positions:
                grid = grid + " t"
            elif (obs_pos, floor_height) in validation_positions:
                grid = grid + " v"
            else:
                grid = grid + " o"
        print(grid)


