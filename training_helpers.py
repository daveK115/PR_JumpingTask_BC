from simplemodel_v3 import SimpleModelV3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import matplotlib.pyplot as plt


def train_model(model, optimizer, criterion, num_epochs, dataloader_train, dataloader_val, validate=False, print_out=False, draw_plot=False):
    losses = []
    val_accuracies = []
    max_val_accuracy = 0.0
    saved_model_loss = float("inf")
    wandb.run.summary["act_epochs"] = num_epochs
    for epoch in range(num_epochs):
        model.train()
        epoch_loss_n = 0
        epoch_loss_sum = 0.0
        for data, target in dataloader_train:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            losses.append(loss.item())
            wandb.log({"loss_train": loss})
            epoch_loss_n += 1
            epoch_loss_sum += loss.item()
            optimizer.step()

        epoch_loss_avg = epoch_loss_sum / epoch_loss_n   
            
        if epoch % 5 == 0:
            if validate:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, targets in dataloader_val:
                        outputs = model(data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
        
                val_accuracy = 100 * correct / total
                wandb.log({"acc_val": val_accuracy})
                val_accuracies.append(val_accuracy)
                
                if print_out:
                    print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy}, Training Loss (Epoch AVG): {epoch_loss_avg}")
                
                if val_accuracy > max_val_accuracy or (val_accuracy == max_val_accuracy and epoch_loss_avg < best_epoch_loss):
                    max_val_accuracy = val_accuracy
                    best_epoch_loss = epoch_loss_avg
                    if print_out:
                        
                        print(f"New best model: Validation Accuracy: {val_accuracy}, Training Loss (Epoch AVG): {epoch_loss_avg}")
    
                    torch.save(model.state_dict(), "best_model.pt")
                    wandb.run.summary["acc_val"] = max_val_accuracy
                    wandb.run.summary["loss_train"] = best_epoch_loss
                
                if val_accuracy == 100.0:
                    wandb.run.summary["act_epochs"] = epoch
                    break
            else :
                if print_out:
                    print(f"Epoch {epoch+1}, Training Loss (Epoch AVG): {epoch_loss_avg}")
    
    if validate:            
        model.load_state_dict(torch.load("best_model.pt"))

    if draw_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Loss')
        plt.title('Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.figure(figsize=(10, 5))
        plt.plot(val_accuracies, label='Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Iterations*10')
        plt.ylabel('Accuracy')
        plt.legend()


# deprecated
def dropout_grid_search(dropout_rates_conv, dropout_rates_fc, dataloader):
    results = {}
    for rate_conv in dropout_rates_conv:
        for rate_fc in dropout_rates_fc:
            model = SimpleModelV3(rate_conv, rate_fc)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

            for epoch in range(200):
                for data, target in dataloader:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    #losses.append(loss.item())
                    optimizer.step()

            print(f'rate_conv: {rate_conv}, rate_fc: {rate_fc}, Loss: {loss.item()}')


# deprecated
def hyperparameter_grid_search(parameter_grid, train_loader, val_loader, validation_interval=10):
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0
    best_params = {}
    results = []

    for lr in parameter_grid["lr"]:
        for weight_decay in parameter_grid["weight_decay"]:
            for dropout_rate_conv in parameter_grid["dropout_rate_conv"]:
                for dropout_rate_fc in parameter_grid["dropout_rate_fc"]:
                    model = SimpleModelV3(dropout_rate_conv=dropout_rate_conv, dropout_rate_fc=dropout_rate_fc)
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                    model_results = []
                    for num_epochs in parameter_grid["epochs"]:
                        model_results.append({"lr": lr, "weight_decay": weight_decay,
                                     "dropout_rate_conv": dropout_rate_conv, "dropout_rate_fc": dropout_rate_fc,
                                     "epochs": num_epochs, "valid_accuracy": []})

                    for epoch in range(np.max(parameter_grid["epochs"])):
                        model.train()
                        for data, target in train_loader:
                            optimizer.zero_grad()
                            output = model(data)
                            loss = criterion(output, target)
                            loss.backward()
                            optimizer.step()

                        if epoch % validation_interval == 0:
                            model.eval()
                            correct = 0
                            total = 0
                            with torch.no_grad():
                                for data, targets in val_loader:
                                    outputs = model(data)
                                    _, predicted = torch.max(outputs.data, 1)
                                    total += targets.size(0)
                                    correct += (predicted == targets).sum().item()
                            accuracy = 100 * correct / total

                            for i in range(len(model_results)):
                                if model_results[i]["epochs"] >= epoch:
                                    model_results[i]["valid_accuracy"].append((epoch, accuracy))

                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_params = model_results
                                # Save model
                                torch.save(model.state_dict(), "best_model.pth")

                            print(f"Epoch {epoch + 1}, LR {lr}, WD {weight_decay}, DRC {dropout_rate_conv}, DRFC {dropout_rate_fc}, Validation Accuracy: {accuracy}")
                            print(f'Epoch {epoch + 1}, Training Loss: {loss.item()}')

                    for i in range(len(model_results)):
                        results.append(model_results[i])
    return results, best_params





