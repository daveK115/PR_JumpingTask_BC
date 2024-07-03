import numpy as np
from jumping_task import JumpTaskEnv
import torch
import random


def generate_optimal_episode(obstacle_position, floor_height):
    """Generates one optimal episode with the specified environment config"""
    env = JumpTaskEnv(scr_w=60, scr_h=60, slow_motion=False, rendering=False)
    state = env._reset(obstacle_position=obstacle_position, floor_height=floor_height)

    terminal = False
    i = 0
    states, rewards, actions = [], [], []

    while not terminal:
        i += 1
        if i == obstacle_position - 13:
            action = 1
        else:
            action = 0

        next_state, reward, terminal, _ = env.step(action)
        states.append(state)
        rewards.append(reward)
        actions.append(action)

        state = next_state

    return states, np.array(actions), np.array(rewards)


def generate_imitation_data():
    """Generates perfect episodes for each environment configuration in the experiment space"""
    imitation_data = {}
    for obs_pos in range(20, 46):
        imitation_data[obs_pos] = {}
        for floor_height in range(10, 21):
            states, actions, rewards = generate_optimal_episode(obs_pos, floor_height)
            states = np.stack(states)
            imitation_data[obs_pos][floor_height] = (states, actions, rewards)
    return imitation_data


def generate_training_positions(min_obstacle_position=20,
                                max_obstacle_position=45,
                                min_floor_height=10,
                                max_floor_height=20,
                                positions_train_diff=5,
                                heights_train_diff=5,
                                random_tasks=False):
    """Generates Training set positions, default parameterr correspond to 'wide' grid"""
    if random_tasks:
        obstacle_positions = list(range(min_obstacle_position, max_obstacle_position + 1))
        floor_heights = list(range(min_floor_height, max_floor_height + 1))
        num_positions = (len(obstacle_positions) // positions_train_diff) + 1
        num_heights = (len(floor_heights) // heights_train_diff) + 1
        num_train_positions = num_positions * num_heights
        obstacle_positions_train = np.random.choice(
            obstacle_positions, size=num_train_positions)
        floor_heights_train = np.random.choice(
            floor_heights, size=num_train_positions)
        training_positions = list(
            zip(obstacle_positions_train, floor_heights_train))
    else:
        obstacle_positions_train = list(
            range(min_obstacle_position, max_obstacle_position + 1,
                  positions_train_diff))
        floor_heights_train = list(
            range(min_floor_height, max_floor_height + 1, heights_train_diff))

        training_positions = []
        for pos in obstacle_positions_train:
            for height in floor_heights_train:
                training_positions.append((pos, height))
    return training_positions


def prepare_observation_target_data(positions, imitation_data):
    observations, targets = [], []
    for pos, height in positions:
        states, actions, _ = imitation_data[pos][height]
        for state in states:
            observations.append(state)
        for action in actions:
            targets.append(action)

    return observations, targets


def generate_augmented_data_horiz(observations, actions, max_shift_left=3, max_shift_right=3):
    """Generates augmented dataset from given dataset, shifting horizontally in pixel space"""
    x_augmented, y_augmented = [], []
    for observation, action in zip(observations, actions):
        x_augmented.append(observation)
        y_augmented.append(action)

        for i in range(1, max_shift_left+1):
            shifted = np.zeros_like(observation)
            shifted[:, i:] = observation[:, :-i]
            x_augmented.append(shifted)
            y_augmented.append(action)

        for i in range(1, max_shift_right+1):
            shifted = np.zeros_like(observation)
            shifted[:, :-i] = observation[:, i:]
            x_augmented.append(shifted)
            y_augmented.append(action)

    return x_augmented, y_augmented


def generate_augmented_data_vert(observations, actions, max_shift_up=3, max_shift_down=3):
    """Generates augmented dataset from given dataset, shifting vertically in pixel space"""
    x_augmented, y_augmented = [], []
    for observation, action in zip(observations, actions):
        x_augmented.append(observation)
        y_augmented.append(action)

        for i in range(1, max_shift_up+1):
            shifted = np.zeros_like(observation)
            shifted[i:, :] = observation[:-i, :]
            x_augmented.append(shifted)
            y_augmented.append(action)

        for i in range(1, max_shift_down+1):
            shifted = np.zeros_like(observation)
            shifted[:-i, :] = observation[i:, :]
            x_augmented.append(shifted)
            y_augmented.append(action)

    return x_augmented, y_augmented


def generate_validation_positions_adjacent(training_positions, min_pos=20, min_height=10, max_pos=45, max_height=20):
    """Generates adjacent positions for validation set based on given training positions"""
    validation_positions = []
    for (obs_pos, floor_height) in training_positions:
        neighbouring_configs = [(obs_pos, floor_height - 1), (obs_pos, floor_height + 1), (obs_pos - 1, floor_height),
                                (obs_pos + 1, floor_height)]
        for (neighbour_obs_pos, neighbour_floor_height) in neighbouring_configs:
            if min_pos <= neighbour_obs_pos <= max_pos and min_height <= neighbour_floor_height <= max_height:
                if not (neighbour_obs_pos, neighbour_floor_height) in validation_positions:
                    validation_positions.append((neighbour_obs_pos, neighbour_floor_height))
    return validation_positions


def generate_validation_positions_random(training_positions, min_pos=20, min_height=10, max_pos=45, max_height=20, n_positions=54):
    """Generates random positions for validation set based on given training positions"""
    validation_positions = []
    for i in range(n_positions):
        found_pos = False

        while not found_pos:
            obs_pos = random.randint(min_pos, max_pos)
            floor_height = random.randint(min_height, max_height)

            if not (obs_pos, floor_height) in training_positions:
                validation_positions.append((obs_pos, floor_height))
                found_pos = True

    return validation_positions


def calculate_sampler_weights(targets):
    """Calculates custom sampler weights based on the given target distribution"""
    class_sample_count = np.array([len(np.where(targets == action)[0]) for action in np.unique(targets)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[action] for action in targets])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    return samples_weight
