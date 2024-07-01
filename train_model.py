import torch
from torch import nn, optim
from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader

from data_helpers import generate_imitation_data, generate_training_positions, generate_validation_positions_adjacent, \
    prepare_observation_target_data, generate_augmented_data_horiz, calculate_sampler_weights
from evaluation_helpers import test_agent
from misc_helpers import setup_seed, get_device
import wandb

from simplemodel_v3 import SimpleModelV3
from training_helpers import train_model


def main():
    setup_seed(seed=42)
    device = get_device()
    imitation_data = generate_imitation_data()

    # Wide grid
    training_positions = generate_training_positions()

    # tight grid
    # training_positions = generate_training_positions(min_obstacle_position=28,
    #                                 max_obstacle_position=38,
    #                                 min_floor_height=13,
    #                                 max_floor_height=17,
    #                                 positions_train_diff=2,
    #                                 heights_train_diff=2)

    # Random Train Split
    # training_positions = generate_training_positions(random_tasks=True)

    validation_positions = generate_validation_positions_adjacent(training_positions, 20, 10, 45, 20)

    x_train, y_train = prepare_observation_target_data(training_positions, imitation_data)
    x_val, y_val = prepare_observation_target_data(validation_positions, imitation_data)

    x_train_augmented, y_train_augmented = generate_augmented_data_horiz(x_train, y_train, 3, 3)
    # x_train_augmented, y_train_augmented = generate_augmented_data_vert(x_train_augmented, y_train_augmented, 2, 2)

    samples_weights_train = calculate_sampler_weights(y_train).to(device)
    sampler_train = WeightedRandomSampler(samples_weights_train, len(samples_weights_train))

    samples_weights_train_augmented = calculate_sampler_weights(y_train_augmented).to(device)
    sampler_train_augmented = WeightedRandomSampler(samples_weights_train_augmented,
                                                    len(samples_weights_train_augmented))

    if len(validation_positions) > 0:
        samples_weights_val = calculate_sampler_weights(y_val)
        sampler_val = WeightedRandomSampler(samples_weights_val, len(samples_weights_val))

    setup_seed(42)

    train_dataset = TensorDataset(torch.Tensor(x_train).unsqueeze(1).to(device),
                                  torch.LongTensor(y_train).to(device))
    train_dataloader = DataLoader(train_dataset, batch_size=64, sampler=sampler_train)

    train_dataset_augmented = TensorDataset(torch.Tensor(x_train_augmented).unsqueeze(1).to(device),
                                            torch.LongTensor(y_train_augmented).to(device))
    train_dataloader_augmented = DataLoader(train_dataset_augmented, batch_size=64, sampler=sampler_train_augmented)

    if len(validation_positions) > 0:
        val_dataset = TensorDataset(torch.Tensor(x_val).unsqueeze(1).to(device),
                                    torch.LongTensor(y_val).to(device))
        val_dataloader = DataLoader(val_dataset, batch_size=64, sampler=sampler_val)
    else:
        val_dataset = None
        val_dataloader = None

    model = SimpleModelV3(dropout_rate_conv=0.3, dropout_rate_fc=0.1)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)

    run = wandb.init(
        project="JumpingTask_BC",
        config={
            "learning_rate": 0.0005,
            "architecture": "SimpleModelV3",
            "dropout_rate_conv": 0.3,
            "dropout_rate_fc": 0.1,
            "weight_decay": 0.0001,
            "dataset": "Wide-Grid-Augmented-3-3",
            "batch_size": 64,
            "validation_positions": "Adjacent",
            "epochs": 400,
        }
    )

    train_model(model, optimizer, criterion, 300, train_dataloader_augmented, val_dataloader, validate=True, print_out=True)
    solved_total, solved_train, solved_val, solved_test = test_agent(model, device, training_positions,
                                                                     validation_positions, True)
    run.summary["solved_envs"] = solved_total
    run.summary["solved_envs_train"] = solved_train
    run.summary["solved_envs_val"] = solved_val
    run.summary["solved_envs_test"] = solved_test
    run.summary["validation_size"] = len(validation_positions)
    run.log_model(path="best_model.pt", name="model")
    run.finish()

    print(f"Solved {solved_total} out of 286 Environments")


if __name__ == "__main__":
    main()
