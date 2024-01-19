from typing import List
from torchvision import transforms, models, datasets
import torch 
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
import numpy as np 
import random 
from os.path import exists
import os 
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, confusion_matrix


def get_oxford_splits(
    batch_size: int,
    data_loader_seed: int = 111, 
    pin_memory: bool = True,
    num_workers: int = 2,
    ): 
    K = 5
    num_support = 80
    num_query = 20 

    def seed_worker(worker_id):
        # worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(data_loader_seed)
        random.seed(data_loader_seed)
    g = torch.Generator()
    g.manual_seed(data_loader_seed)

    support_classes = list(np.arange(num_support))
    query_classes = list(np.arange(num_query) + num_support)

    img_dim = 64

    train_transforms = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    data_path = f'/content/data'

    train_ds_full = datasets.Flowers102(root=data_path, split="train", download=True, transform=train_transforms)
    val_ds_full = datasets.Flowers102(root=data_path, split="val", download=True, transform=validation_transforms)
    test_ds_full = datasets.Flowers102(root=data_path, split="test", download=True, transform=test_transforms)

    train_indxs_support = torch.where(torch.isin(torch.tensor(train_ds_full._labels), torch.asarray(support_classes)))[0]
    val_indxs_support = torch.where(torch.isin(torch.tensor(val_ds_full._labels), torch.asarray(support_classes)))[0]
    test_indxs_support = torch.where(torch.isin(torch.tensor(test_ds_full._labels), torch.asarray(support_classes)))[0]
    
    train_ds_subset_support = torch.utils.data.Subset(train_ds_full, train_indxs_support)
    val_ds_subset_support = torch.utils.data.Subset(val_ds_full, val_indxs_support)
    test_ds_subset_support = torch.utils.data.Subset(test_ds_full, test_indxs_support)

    merged_dataset = ConcatDataset([train_ds_subset_support, val_ds_subset_support, test_ds_subset_support])
    ### A, B
    train_ds_support, test_ds_support = torch.utils.data.random_split(merged_dataset, [0.75, 0.25], generator=torch.Generator().manual_seed(42))
    ### 

    train_indxs_query = torch.where(torch.isin(torch.tensor(train_ds_full._labels), torch.asarray(query_classes)))[0]
    N = 10 
    starting_indices = np.arange(0, len(train_indxs_query), N)
    train_indxs_query = np.hstack([train_indxs_query[i:i+K] for i in starting_indices if i + K <= len(train_indxs_query)])
    ### C
    train_ds_query = torch.utils.data.Subset(train_ds_full, train_indxs_query) 
    ###

    val_indxs_query = torch.where(torch.isin(torch.tensor(val_ds_full._labels), torch.asarray(query_classes)))[0]
    test_indxs_query = torch.where(torch.isin(torch.tensor(test_ds_full._labels), torch.asarray(query_classes)))[0]
    val_ds_subset_query = torch.utils.data.Subset(val_ds_full, val_indxs_query)
    test_ds_subset_query = torch.utils.data.Subset(test_ds_full, test_indxs_query)
    
    test_ds_query_full = ConcatDataset([val_ds_subset_query, test_ds_subset_query])
    ### D 
    _, test_ds_query = torch.utils.data.random_split(test_ds_query_full, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
    ###
    
    ### E 
    test_all = ConcatDataset([test_ds_query, test_ds_support])


    A_train_dl = DataLoader(
        train_ds_support,
        batch_size = batch_size, 
        shuffle=True, 
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    A_test_dl = DataLoader(
        test_ds_support,
        batch_size = batch_size, 
        shuffle=True, 
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers
    )

    B_train_dl = DataLoader(
        train_ds_query,
        batch_size = batch_size, 
        shuffle=True, 
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    B_test_dl = DataLoader(
        test_ds_query,
        batch_size = batch_size, 
        shuffle=True, 
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    test_all = DataLoader(
        test_all,
        batch_size = batch_size, 
        shuffle=True, 
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    
    return A_train_dl, A_test_dl, B_train_dl, B_test_dl, test_all


def make_dir(dir_name: str):
    """
    creates directory "dir_name" if it doesn't exists
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def custom_plot_training_stats(
        acc_hist, 
        loss_hist, 
        phase_list, 
        title: str, 
        dir: str, 
        name: str = 'acc_loss'): 
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize=[14, 6], dpi=100)

    for phase in phase_list: 
        lowest_loss_x = np.argmin(np.array(loss_hist[phase]))
        lowest_loss_y = loss_hist[phase][lowest_loss_x]
        
        ax1.annotate("{:.4f}".format(lowest_loss_y), [lowest_loss_x, lowest_loss_y])
        ax1.plot(loss_hist[phase], '-x', label=f'{phase} loss', markevery = [lowest_loss_x])

        ax1.set_xlabel(xlabel='epochs')
        ax1.set_ylabel(ylabel='loss')

        ax1.grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        ax1.legend()
        ax1.label_outer()

    # acc: 
    for phase in phase_list:
        highest_acc_x = np.argmax(np.array(acc_hist[phase]))
        highest_acc_y = acc_hist[phase][highest_acc_x]
        
        ax2.annotate("{:.4f}".format(highest_acc_y), [highest_acc_x, highest_acc_y])
        ax2.plot(acc_hist[phase], '-x', label=f'{phase} acc', markevery = [highest_acc_x])

        ax2.set_xlabel(xlabel='epochs')
        ax2.set_ylabel(ylabel='acc')

        ax2.grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        ax2.legend()
        #ax2.label_outer()

    fig.suptitle(f'{title}')

    make_dir(dir)
    plt.savefig(f'{dir}/{name}.jpg')
    plt.clf()

def plot_conf(labels, preds, title, dir_, name):
    """
    labels: an [N, ] array containing true labels for N samples
    preds: an [N, ] array containing predications for N samples
    
    saves confusion matrix plot of the given prediction and true labels in 'dir_/name.jpg' 
    """

    conf = confusion_matrix(labels, preds)

    plt.clf()
    cm = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
    cmap = sns.light_palette("navy", as_cmap=True)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, cmap=cmap, fmt=".2f", cbar=False)
    plt.title(f'{title}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    make_dir(dir_)
    plt.savefig(f'{dir_}/{name}')
