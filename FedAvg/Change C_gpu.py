import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import itertools
import copy
import random
import time

device = torch.device("cpu")  # 强制使用 CPU

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 12800)
        self.fc2 = nn.Linear(12800, 1280)
        self.fc3 = nn.Linear(1280, 128)
        self.fc4 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
def get_data_loaders(batch_size, num_clients):
    time_start = time.time()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    client_datasets = random_split(dataset, [len(dataset) // num_clients] * num_clients)
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    print(f"Data loading complete, time: {time.time() - time_start}")
    return client_loaders

def client_update(client_model, optimizer, train_loader, epochs):
    client_model.train()
    epoch_losses = []
    for epoch in range(epochs):
        time_start = time.time()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        epoch_losses.append(average_loss)
        print(f"Client training epoch {epoch + 1} complete, average loss: {average_loss}, time: {time.time() - time_start}")
    return epoch_losses

def average_weights(global_model, client_models):
    time_start = time.time()
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)  # 更新全局模型参数

def plot_losses(federated_losses, direct_losses):
    num_rounds = len(federated_losses)
    num_clients = len(federated_losses[0])
    
    avg_losses = []
    avg_accuracies = []
    for round in range(num_rounds):
        round_losses = federated_losses[round]
        flat_round_losses = list(itertools.chain(*round_losses))
        avg_loss = sum(flat_round_losses) / len(flat_round_losses)
        avg_accuracy = 1 - avg_loss
        avg_accuracies.append(avg_accuracy)
        avg_losses.append(avg_loss)
        plt.scatter(round, 1 - avg_loss, color='royalblue', marker='o')

    plt.plot(range(num_rounds), avg_accuracies, color='lightblue', linestyle='-', label='Federated Learning')

    for i in range(len(direct_losses)):
        direct_losses[i] = 1 - direct_losses[i]
    plt.plot(direct_losses, label='Direct Training', linestyle='-', color='black')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')

    plt.axhline(y=0.99, color='grey', linestyle='--', label='y=0.99')

    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

def direct_training(epochs=20):
    batch_size = 64
    learning_rate = 0.01

    model = SimpleNN().to(device)
    train_loader = get_data_loaders(batch_size, 1)[0]  # 获取单个数据加载器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        time_start = time.time()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        losses.append(average_loss)
        print(f"Direct training epoch {epoch + 1} complete, average loss: {average_loss}, time: {time.time() - time_start}")
    return losses

# def federated_learning(rounds=20):
#     num_clients = 10
#     batch_size = 64
#     learning_rate = 0.01
#     epoch = 20
#     C = 0.1

#     global_model = SimpleNN().to(device)
#     client_loaders = get_data_loaders(batch_size, num_clients)
#     all_client_losses = []

#     for round in range(rounds):
#         client_models = [SimpleNN().to(device) for _ in range(num_clients)]
#         for client_model in client_models:
#             client_model.load_state_dict(global_model.state_dict())
        
#         optimizer = [optim.SGD(model.parameters(), lr=learning_rate) for model in client_models]
#         round_losses = []

#         m = max(int(C * num_clients), 1)  # 至少选择一个客户端
#         selected_clients = random.sample(range(num_clients), m)

#         client_losses = []
#         for i in selected_clients:
#             client_losses = client_update(client_models[i], optimizer[i], client_loaders[i], epoch)
#             round_losses.append(client_losses)
        
#         average_weights(global_model, [client_models[i] for i in selected_clients])
#         print(f"Federated Learning round {round + 1} complete, average loss: {sum(client_losses) / len(client_losses)}")
#         all_client_losses.append(round_losses)
    
#     return all_client_losses

def main():
    rounds = 500

    start_time = time.time()
    # federated_losses = federated_learning(rounds)
    direct_losses = direct_training(rounds)
    print("Time elapsed: ", time.time() - start_time)
    # plot_losses(federated_losses, direct_losses)

if __name__ == "__main__":
    main()