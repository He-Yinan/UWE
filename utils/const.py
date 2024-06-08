from torchvision import transforms


budget = {
    'exp05k': {
        'lb': [500, 1000, 2000, 4000, 8000, 16000],
        'vd': 0
    },
    'exp1k': {
        'lb': [1000, 2000, 4000, 8000],
        'vd': 0
    },
    'exp5k': {
        'lb': [5000, 6000, 8000, 12000, 20000, 28000],
        'vd': 0
    }
}

dataset = {
    'cifar10': {
        'tr': 50000,
        'vd': 0,
        'te': 10000,
        'n_channels': 3,
        'n_classes': 10,
        'transform': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ]),
        'normalize': transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
        'loader_tr_args': {
            'batch_size': 256,
            'num_workers': 4
        },
        'loader_te_args': {
            'batch_size': 256,
            'num_workers': 4
        },
    },
    'cifar100': {
        'tr': 50000,
        'vd': 0,
        'te': 10000,
        'n_channels': 3,
        'n_classes': 100,
        'transform': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ]),
        'normalize': transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
        'loader_tr_args': {
            'batch_size': 256,
            'num_workers': 4
        },
        'loader_te_args': {
            'batch_size': 256,
            'num_workers': 4
        },
    },
    'mini_imagenet': {
        'tr': 48000,
        'vd': 0,
        'te': 12000,
        'n_channels': 3,
        'n_classes': 100,
        'transform': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        'normalize': transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'loader_tr_args': {
            'batch_size': 64,
            'num_workers': 4
        },
        'loader_te_args': {
            'batch_size': 256,
            'num_workers': 4
        },
    }
}
