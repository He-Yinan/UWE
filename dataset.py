import os
import pickle

import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset

from utils import logger, getter, read_write


def get_dataset(args):
    l = logger.Logger.get(args)

    if args.dataset == 'cifar10':
        data = get_cifar10()
    elif args.dataset == 'cifar100':
        data = get_cifar100()
    elif args.dataset == 'mini_imagenet':
        data = get_mini_imagenet()
    else:
        raise NotImplementedError
    
    x_tr = data['x_tr']
    y_tr = data['y_tr']
    x_te = data['x_te']
    y_te = data['y_te']

    if args.action == 'supervised':
        lb_idxs = np.arange(len(y_tr), dtype=int)
        vd_idxs = np.arange(0, dtype=int)
        exclude_vd = np.ones(len(y_tr), dtype=bool)
        lb_flag = np.zeros(len(y_tr), dtype=bool)
    elif args.action != 'initialize':
        folder = getter.get_init_folder(args)
        lb_idxs_file = getter.get_init_file(args, 'init_lb')
        vd_idxs_file = getter.get_init_file(args, 'vd')
        lb_idxs = np.array(read_write.read_csv(os.path.join(folder, lb_idxs_file)), dtype=int)
        vd_idxs = np.array(read_write.read_csv(os.path.join(folder, vd_idxs_file)), dtype=int)
        exclude_vd = np.ones(len(y_tr), dtype=bool)
        lb_flag = np.zeros(len(y_tr), dtype=bool)

    if len(vd_idxs) > 0:
        exclude_vd[vd_idxs] = False
    if len(lb_idxs) > 0:
        lb_flag[lb_idxs] = True
    lb_flag = lb_flag[exclude_vd]

    x_tr_exclude_vd, x_vd = x_tr[exclude_vd], x_tr[vd_idxs]
    y_tr_exclude_vd, y_vd = y_tr[exclude_vd], y_tr[vd_idxs]

    l.info(f'lb_flag: {lb_flag.shape}, n_labeled: {np.sum(lb_flag)}')
    l.info(f'x_tr: {x_tr.shape}, x_tr_exclude_vd: {x_tr_exclude_vd.shape}, x_te: {x_te.shape}, x_vd: {x_vd.shape}')
    l.info(f'y_tr: {y_tr.shape}, y_tr_exclude_vd: {y_tr_exclude_vd.shape}, y_te: {y_te.shape}, y_vd: {y_vd.shape}')

    return {
        'x_tr': x_tr_exclude_vd,
        'x_vd': x_vd,
        'x_te': x_te,
        'y_tr': y_tr_exclude_vd,
        'y_vd': y_vd,
        'y_te': y_te,
        'lb_flag': lb_flag
    }    


def get_handler(dataset):
    if dataset == 'cifar10':
        return CifarHandler
    elif dataset == 'cifar100':
        return CifarHandler
    elif dataset == 'mini_imagenet':
        return ImagenetHandler
    else:
        raise NotImplementedError


def get_cifar10():
    raw_tr = datasets.CIFAR10('./dataset/CIFAR10', train=True, download=True)
    raw_te = datasets.CIFAR10('./dataset/CIFAR10', train=False, download=True)
    
    return {
        'x_tr': raw_tr.data,
        'y_tr': torch.from_numpy(np.array(raw_tr.targets)),
        'x_te': raw_te.data,
        'y_te': torch.from_numpy(np.array(raw_te.targets))
    }

def get_cifar100():
    raw_tr = datasets.CIFAR100('./dataset/CIFAR100', train=True, download=True)
    raw_te = datasets.CIFAR100('./dataset/CIFAR100', train=False, download=True)

    return {
        'x_tr': raw_tr.data,
        'y_tr': torch.from_numpy(np.array(raw_tr.targets)),
        'x_te': raw_te.data,
        'y_te': torch.from_numpy(np.array(raw_te.targets))
    }


def get_mini_imagenet():
    f = open(os.path.join('dataset', 'miniimagenet', 'mini-imagenet-cache-train.pkl'), 'rb')
    train_data = pickle.load(f)
    f = open(os.path.join('dataset', 'miniimagenet', 'mini-imagenet-cache-val.pkl'), 'rb')
    val_data = pickle.load(f)
    f = open(os.path.join('dataset', 'miniimagenet', 'mini-imagenet-cache-test.pkl'), 'rb')
    test_data = pickle.load(f)

    labels = list(train_data['class_dict'].keys()) + list(val_data['class_dict'].keys()) + list(test_data['class_dict'].keys())
    image_count = len(train_data['class_dict'][labels[0]])
    test_proportion = int(image_count * 0.2)
    train_proportion = image_count - test_proportion

    image_dim = train_data['image_data'].shape[1]
    X_tr = np.zeros((len(labels) * train_proportion, image_dim, image_dim, 3), dtype=np.uint8)
    Y_tr = torch.ones((len(labels) * train_proportion), dtype=torch.long)
    X_te = np.zeros((len(labels) * test_proportion, image_dim, image_dim, 3), dtype=np.uint8)
    Y_te = torch.ones((len(labels) * test_proportion), dtype=torch.long)

    idx = 0
    for label in train_data['class_dict']:
        X_te[idx * test_proportion:(idx + 1) * test_proportion] = train_data['image_data'][train_data['class_dict'][label][:test_proportion]]
        Y_te[idx * test_proportion:(idx + 1) * test_proportion] *= labels.index(label)

        X_tr[idx * train_proportion:(idx + 1) * train_proportion] = train_data['image_data'][train_data['class_dict'][label][test_proportion:]]
        Y_tr[idx * train_proportion:(idx + 1) * train_proportion] *= labels.index(label)

        idx += 1

    for label in val_data['class_dict']:
        X_te[idx * test_proportion:(idx + 1) * test_proportion] = val_data['image_data'][val_data['class_dict'][label][:test_proportion]]
        Y_te[idx * test_proportion:(idx + 1) * test_proportion] *= labels.index(label)

        X_tr[idx * train_proportion:(idx + 1) * train_proportion] = val_data['image_data'][val_data['class_dict'][label][test_proportion:]]
        Y_tr[idx * train_proportion:(idx + 1) * train_proportion] *= labels.index(label)

        idx += 1

    for label in test_data['class_dict']:
        X_te[idx * test_proportion:(idx + 1) * test_proportion] = test_data['image_data'][test_data['class_dict'][label][:test_proportion]]
        Y_te[idx * test_proportion:(idx + 1) * test_proportion] *= labels.index(label)

        X_tr[idx * train_proportion:(idx + 1) * train_proportion] = test_data['image_data'][test_data['class_dict'][label][test_proportion:]]
        Y_tr[idx * train_proportion:(idx + 1) * train_proportion] *= labels.index(label)

        idx += 1

    return {
        'x_tr': X_tr,
        'y_tr': Y_tr,
        'x_te': X_te,
        'y_te': Y_te
    }


class CifarHandler(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.x)


class ImagenetHandler(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.x)
