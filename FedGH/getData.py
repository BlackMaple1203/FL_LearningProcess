import random
from collections import defaultdict
from torch.utils.data import random_split
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

def get_data(data_name, dataroot, normalize=True, val_size=10000):
    """
    get_datasets returns train/val/test data splits of CIFAR10/100 datasets
    :param data_name: name of dataset, choose from [cifar10, cifar100]
    :param dataroot: root to data dir
    :param normalize: True/False to normalize the data
    :param val_size: validation split size (in #samples)
    :return: train_set, val_set, test_set (tuple of pytorch dataset/subset)
    """

    if data_name =='CIFAR10':
        norm_mean = [0.485, 0.456, 0.406] 
        norm_std = [0.229, 0.224, 0.225]  
        data_obj = CIFAR10
    elif data_name == 'CIFAR100':
        norm_mean = [0.5071, 0.4865, 0.4409]  
        norm_std = [0.2673, 0.2564, 0.2762]  
        data_obj = CIFAR100
    elif data_name == 'MNIST':
        norm_mean = [0.5071]  
        norm_std = [0.2673]  
        data_obj = MNIST
    else:
        raise ValueError("choose data_name from ['MNIST', 'CIFAR10', 'CIFAR100']")

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),  
            transforms.RandomHorizontalFlip(),  
            transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  
            transforms.RandomCrop(32, padding=4)  
        ])
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),  
        ])
    dataset = data_obj(
        dataroot,
        train=True,
        download=True,
        transform=transform_train
    )
    test_set = data_obj(
        dataroot,
        train=False,
        download=True,
        transform=transform_test
    )
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_set, val_set, test_set
    
def gen_random_loaders(data_name, data_path, num_users, bz, classes_per_user):
    """
    generates train/val/test loaders of each client
    :param data_name: name of dataset, choose from [cifar10, cifar100]
    :param data_path: root path for data dir
    :param num_users: number of clients
    :param bz: batch size
    :param classes_per_user: number of classes assigned to each client
    :return: train/val/test loaders of each client, list of pytorch dataloaders
    """
    loader_params = {"batch_size": bz, "shuffle": False, "pin_memory": True, "num_workers": 0}
    dataloaders = []
    datasets = get_data(data_name, data_path) # datasets: train_set, val_set, test_set
    for i, d in enumerate(datasets):
        # ensure same partition for train/test/val
        if i == 0: # train set
            cls_partitions = gen_classes_per_node(d, num_users, classes_per_user)
            loader_params['shuffle'] = True
        usr_subset_idx = gen_data_split(d, num_users, cls_partitions)

        if i == 0:
            train_sample_count = [len(i) for i in usr_subset_idx]
        elif i == 1:
            eval_sample_count = [len(i) for i in usr_subset_idx]
        else:
            test_sample_count = [len(i) for i in usr_subset_idx]
        # create subsets for each client
        subsets = list(map(lambda x: torch.utils.data.Subset(d, x), usr_subset_idx))
        # create dataloaders from subsets
        dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))

    return dataloaders[0], dataloaders[1], dataloaders[2], train_sample_count, eval_sample_count, test_sample_count

def gen_data_split(dataset, num_users, class_partitions):
    """
    divide data indexes for each client based on class_partition
    :param dataset: pytorch dataset object (train/val/test)
    :param num_users: number of clients
    :param class_partitions: proportion of classes per client
    :return: dictionary mapping client to its indexes
    """
    num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset)

    # -------------------------- #
    # Create class index mapping #
    # -------------------------- #
    data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}

    # --------- #
    # Shuffling #
    # --------- #
    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)

    # ------------------------------ #
    # Assigning samples to each user #
    # ------------------------------ #
    user_data_idx = [[] for i in range(num_users)]
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    return user_data_idx

def gen_classes_per_node(dataset, num_users, classes_per_user=2, high_prob=0.6, low_prob=0.4): #high_prob=0.6, low_prob=0.4
    """
    creates the data distribution of each client
    :param dataset: pytorch dataset object
    :param num_users: number of clients
    :param classes_per_user: number of classes assigned to each client
    :param high_prob: highest prob sampled
    :param low_prob: lowest prob sampled
    :return: dictionary mapping between classes and proportions, each entry refers to other client
    """
    num_classes, num_samples, _ = get_num_classes_samples(dataset)

    # -------------------------------------------#
    # Divide classes + num samples for each user #
    # -------------------------------------------#
    # assert (classes_per_user * num_users) % num_classes == 0, "equal classes appearance is needed" 
    count_per_class = (classes_per_user * num_users) // num_classes
    class_dict = {}
    for i in range(num_classes): 
        # sampling alpha_i_c
        probs = np.random.uniform(low_prob, high_prob, size=count_per_class)
        # normalizing
        probs_norm = (probs / probs.sum()).tolist()
        # probs_norm = [1 / count_per_class] * count_per_class  # balanced
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

    # -------------------------------------#
    # Assign each client with data indexes #
    # -------------------------------------#
    class_partitions = defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]['count'] for i in range(num_classes)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
    return class_partitions

def get_num_classes_samples(dataset):
    """
    extracts info about certain dataset
    :param dataset: pytorch dataset object
    :return: dataset info number of classes, number of samples, list of labels
    """
    # ---------------#
    # Extract labels #
    # ---------------#
    if isinstance(dataset, torch.utils.data.Subset):
        if isinstance(dataset.dataset.targets, list):
            data_labels_list = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            data_labels_list = dataset.dataset.targets[dataset.indices]
    else:
        if isinstance(dataset.targets, list):
            data_labels_list = np.array(dataset.targets)
        else:
            data_labels_list = dataset.targets
    classes, num_samples = np.unique(data_labels_list, return_counts=True)
    num_classes = len(classes)
    return num_classes, num_samples, data_labels_list