import tools
import math
import copy
import torch
from torch import nn
import numpy as np
import time
import random
import argparse 
import func
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from collections import OrderedDict, defaultdict
from pathlib import Path
from nodes import BaseNodes
from CNN import CNN
from FC import FC

random.seed(2022)

def test_acc(net, testloader,criteria,device):
    net.eval()
    with torch.no_grad():
        test_acc = 0
        num_batch = 0

        for batch in testloader:
            num_batch += 1
            # batch = next(iter(testloader))
            img, label = tuple(t.to(device) for t in batch)
            pred, _ = net(img)
            test_loss = criteria(pred, label)
            test_acc += pred.argmax(1).eq(label).sum().item() / len(label)
        mean_test_loss = test_loss / num_batch
        mean_test_acc = test_acc / num_batch
    return mean_test_loss, mean_test_acc


def train(
        data_name: str,
        date_path: str,
        classes_per_node: int,
        node_nums: int,
        fraction: float,
        steps: int,
        epochs: int,
        optim: str,
        lr: float,
        inner_lr: float,
        embed_lr: float,
        wd: float,
        inner_wd: float,
        embed_dim: int,
        hyper_hid: int,
        n_hidden: int,
        n_kernals: int,
        bs: int,
        device,
        eval_every: int,
        save_path: Path,
        seed: int,
        total_classes: int,
) -> None :
    nodes = BaseNodes(data_name, date_path, node_nums, class_per_node = classes_per_node, batch_size = bs)

    train_sample_count = nodes.train_sample_count
    eval_sample_count = nodes.eval_sample_count
    test_sample_count = nodes.test_sample_count

    client_sample_count = [train_sample_count[i] + eval_sample_count[i] for i in range(len(train_sample_count))]

    print("data_name is: ", data_name)
    if data_name == "CIFAR10":
        net = CNN(n_kernals = n_kernals)
        net_FC = FC()
    elif data_name == "CIFAR100":
        net = CNN(n_kernals = n_kernals, out_dim = 100)
        net_FC = FC(out_dim = 100)
    elif data_name == "MNIST":
        net = CNN(n_kernals = n_kernals)
    else:
        raise ValueError("data_name should be one of ['CIFAR10', 'CIFAR100', 'MNIST']")

    net = net.to(device)
    net_FC = net_FC.to(device)

    init_General_params = copy.deepcopy(net.state_dict())

    optimizers = {
        'sgd': torch.optim.SGD(params=net.parameters(), lr=lr, momentum=0.9, weight_decay=wd),
        'adam': torch.optim.Adam(params=net.parameters(), lr=lr)
    }
    optimizer = optimizers[optim]
    criteria = nn.CrossEntropyLoss()
    
    PM_acc = defaultdict()
    PMs = defaultdict()
    Data_Distributions = defaultdict()
    Protos = defaultdict()
    Global_Proto = defaultdict()
    Global_header = defaultdict()

    for i in range(node_nums):
        PM_acc[i] = 0
        PMs[i] = init_General_params
        Protos[i] = defaultdict(list)
        Data_Distributions[i] = defaultdict(list)
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for step in range(steps):
        round_id = step
        frac = fraction
        select_nodes = random.sample(range(node_nums),int(frac * node_nums))

        loc_train_loss = []
        loc_train_acc = []
        global_train_loss = []
        global_train_acc = []
        
        results = []

        Protos_Mean = defaultdict()
        for i in select_nodes:
            Protos_Mean[i] = defaultdict(list)
        
        print("Round: ", round_id)
        for c in select_nodes:
            node_id = c
            print("Current client id: ", node_id)

            if round_id == 0:
                net.load_state_dict(init_General_params)
            else:
                net_paras = dict(PMs[node_id],**Global_header)
                net.load_state_dict(net_paras)
            
            global_train_loss.append(0)
            global_train_acc.append(0)

            net.train()
            for i in range(epochs):
                for j,batch in enumerate(nodes.train_loaders[node_id],0):
                    img, label = tuple(t.to(device) for t in batch)
                    optimizer.zero_grad()
                    pred, rep = net(img)
                    loss = criteria(pred, label)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 50)
                    optimizer.step()
            
            full_net = copy.deepcopy(net.state_dict())
            del_keys = list(full_net.keys())[-2:]
            for key in del_keys:
                full_net.pop(key)
            
            PMs[node_id] = copy.deepcopy(full_net)
            trained_loss, trained_acc = test_acc(net, nodes.train_loaders[node_id], criteria, device)
            loc_train_loss.append(trained_loss.cpu().item())
            loc_train_acc.append(trained_acc)
            PM_acc[node_id] = trained_acc

            print(f"Round {round_id}, Client {node_id}, Train Loss: {trained_loss}, Train Acc: {trained_acc}")

            proto_mean = defaultdict(list)

            for j,batch in enumerate(nodes.train_loaders[node_id],0):
                img, label = tuple(t.to(device) for t in batch)
                pred, rep = net(img)

                owned_classes = label.unique().detach().cpu().numpy()
                for cls in owned_classes:
                    filted_reps = list(map(lambda x: x[0], filter(lambda x: x[1] == cls, zip(rep, label))))
                    sum_filted_reps = filted_reps[0].detach()
                    for f in range(1,len(filted_reps)):
                        sum_filted_reps += filted_reps[f].detach()
                    
                    mean_filted_reps = sum_filted_reps / len(filted_reps)
                    proto_mean[cls].append(mean_filted_reps)
            
            for cls,protos in proto_mean.items():
                sum_proto = protos[0]
                for m in range(1,len(protos)):
                    sum_proto += protos[m]
                mean_proto = sum_proto / len(protos)

                Protos[node_id][cls] = mean_proto
        
        mean_trained_loss = round(np.mean(loc_train_loss),4)
        mean_trained_acc = round(np.mean(loc_train_acc),4)
        mean_global_loss = round(np.mean(global_train_loss),4)
        mean_global_acc = round(np.mean(global_train_acc),4)
        results.append([mean_trained_loss, mean_trained_acc, mean_global_loss, mean_global_acc]+[round(i,4) for i in PM_acc.values()])

        print(f"Round {round_id}, Mean Train Loss: {mean_trained_loss}, Mean Train Acc: {mean_trained_acc}")

        net_FC.train()
        for c in select_nodes:
            for cls, rep in Protos_Mean[c].items():
                pred_server = net_FC(rep)
                loss = criteria(pred_server.view(1,-1),torch.tensor(cls).view(1).to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_FC.parameters(), 50)
                optimizer.step()
        
        Global_header = copy.deepcopy(net_FC.state_dict())

        print(f"Round {round_id}, Global header updated")
    
    print("Training is finished.")
                
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Federated Learning with Lookahead experiment"
    )

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="CIFAR100", choices=['CIFAR10', 'CIFAR100', 'MNIST'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--total-classes", type=str, default=100)
    parser.add_argument("--data-path", type=str, default="/Users/tony/Desktop/Tony/College/Lab/FedGH/data", help="dir path for MNIST dataset")
    parser.add_argument("--num-nodes", type=int, default=100, help="number of simulated nodes")
    parser.add_argument("--fraction", type=int, default=0.1, help="number of sampled nodes in each round")

    ##################################
    #       Optimization args        #
    ##################################

    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10, help="number of inner steps")

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-3, help="inner weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="Results/temp", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.data_name == 'cifar10':
        args.classes_per_node = 2
    elif args.data_name == 'cifar100':
        args.classes_per_node = 10
    else:
        args.classes_per_node = 2
    
    train(
        data_name=args.data_name,
        date_path=args.data_path,
        classes_per_node=args.classes_per_node,
        total_classes=args.total_classes,
        node_nums=args.num_nodes,
        fraction=args.fraction,
        steps=args.num_steps,
        epochs=args.epochs,
        optim=args.optim,
        lr=args.lr,
        inner_lr=args.inner_lr,
        embed_lr=args.embed_lr,
        wd=args.wd,
        inner_wd=args.inner_wd,
        embed_dim=args.embed_dim,
        hyper_hid=args.hyper_hid,
        n_hidden=args.n_hidden,
        n_kernals=args.nkernels,
        bs=args.batch_size,
        device=device,
        eval_every=args.eval_every,
        save_path=args.save_path,
        seed=args.seed,
    )