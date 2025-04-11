import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import itertools
import copy
import random
import time

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128,10)
    
    def forward(self,x):
        x = x.view(-1,28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def get_data_loaders(batch_size,num_clients):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)

    client_datasets = random_split(dataset, [len(dataset) // num_clients] * num_clients)
    client_loaders = [DataLoader(ds,batch_size=batch_size,shuffle=True) for ds in client_datasets]
    return client_loaders

def client_update(client_model,optimizer,train_loader,epochs):
    client_model.train()
    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for data,target in train_loader:
            optimizer.zero_grad()
            output = client_model(data)
            loss = nn.CrossEntropyLoss()(output,target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss/len(train_loader)
        epoch_losses.append(average_loss)
    return epoch_losses

def average_weights(global_model,client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))],0).mean(0)
    global_model.load_state_dict(global_dict)  # 更新全局模型参数

def plot_losses(federated_losses,C,line_color):
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
        # plt.scatter(round,1-avg_loss, color='royalblue', marker='o')


    plt.plot(range(num_rounds), avg_accuracies, color=line_color, linestyle='-', label='Federated Learning with C = ' + str(C))


    # for i in range(len(direct_losses)):
        # direct_losses[i] = 1 - direct_losses[i]
    # 绘制直接训练的损失曲线
    # plt.plot(direct_losses, label='Direct Training', linestyle='-', color='black')
    
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Training Accuracy Over Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Over Epochs')

    # plt.axhline(y=0.99, color='grey', linestyle='--', label='y=0.99')

    # plt.legend()
    # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.show()

def plot_direct_losses(direct_losses):
    for i in range(len(direct_losses)):
        direct_losses[i] = 1 - direct_losses[i]
    # 绘制直接训练的损失曲线
    plt.plot(direct_losses, label='Direct Training', linestyle='-')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Over Epochs')

    plt.axhline(y=0.99, linestyle='--', label='y=0.99')

    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))




def direct_training(epochs=20,batch_size=64,learning_rate=0.01):

    model = SimpleNN()
    train_loader = get_data_loaders(batch_size, 1)[0]  # 获取单个数据加载器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        losses.append(average_loss)
        print(f"SGD training epoch {epoch+1} comlete, average loss:          {average_loss}")

    return losses

def federated_learning(rounds=20,num_clients = 10,batch_size = 64,learning_rate = 0.01,epoch = 20,C = 0.1):
    global_model = SimpleNN()
    client_loaders = get_data_loaders(batch_size, num_clients)
    all_client_losses = []

    for round in range(rounds):
        client_models = [SimpleNN() for _ in range(num_clients)]
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())
        
        optimizer = [optim.SGD(model.parameters(), lr=learning_rate) for model in client_models]
        round_losses = []

        # 随机选择参与训练的客户端
        m = max(int(C * num_clients), 1)  # 至少选择一个客户端
        selected_clients = random.sample(range(num_clients), m)

        client_losses = []
        for i in selected_clients:
            client_losses = client_update(client_models[i], optimizer[i], client_loaders[i], epoch)
            round_losses.append(client_losses)
        
        average_weights(global_model, [client_models[i] for i in selected_clients])
        print(f"Federated Learning round {round+1} complete, average loss:   {sum(client_losses)/len(client_losses)}")
        all_client_losses.append(round_losses)
    
    return all_client_losses

def main():
    rounds = 3
    C_values = [0.0, 0.1, 0.2, 0.5, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(C_values)))
    start_time = time.time()
    for i, C in enumerate(C_values):
        print(f"Running Federated Learning with C = {C}")
        federated_losses = federated_learning(rounds, C=C)
        plot_losses(federated_losses, C, colors[i])
    direct_losses = direct_training(rounds)
    plot_direct_losses(direct_losses)
    print("Time elapsed: ", time.time() - start_time)
    plt.show()

if __name__ == "__main__":
    main()