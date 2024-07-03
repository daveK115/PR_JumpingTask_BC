import torch

from jumping_task import JumpTaskEnv


def test_agent(model, device, training_positions, validation_positions=[], print_out=False):
    """Test given model on all environments"""
    model.eval()
    env = JumpTaskEnv(scr_w=60, scr_h=60, slow_motion=False, rendering=False)
    completed_total = 0
    completed_train = 0
    completed_val = 0
    completed_test = 0

    total = 0
    for floor_height in range(10, 21):
        grid = ''
        for obs_pos in range(20, 46):
            total += 1
            total_reward = 0
            state = env._reset(obstacle_position=obs_pos, floor_height=floor_height)
            terminal = False
            while not terminal:
                state_tensor = torch.Tensor(state).unsqueeze(0).unsqueeze(0).to(device)
                output = model(state_tensor).cpu().detach().numpy()
                if output[0][0] > output[0][1]:
                    action = 0
                else:
                    action = 1

                state, reward, terminal, _ = env.step(action)
                total_reward += reward

            if total_reward > 100:
                if (obs_pos, floor_height) in training_positions:
                    completed_train += 1
                    grid = grid + " t"
                elif (obs_pos, floor_height) in validation_positions:
                    completed_val += 1
                    grid = grid + " x"
                else:
                    completed_test += 1
                    grid = grid + " x"

                completed_total += 1
            else:
                if (obs_pos, floor_height) in training_positions:
                    grid = grid + " t"
                else:
                    grid = grid + " o"
        if print_out:
            print(grid)
    if print_out:
        print(f"completed {completed_total} out of {total} test games")

    return completed_total, completed_train, completed_val, completed_test
