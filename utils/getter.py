import os
from pathlib import Path

from utils import const
from model import resnet, vision_transformer
from al_strategies import Random, Entropy, UWE, Coreset, BADGE, Bald, Bvsb, LeastConfident


def get_strategy(args):
    if args.method == 'random':
        return Random(args)
    elif args.method == 'entropy':
        return Entropy(args)
    elif args.method == 'bvsb':
        return Bvsb(args)
    elif args.method == 'least_confident':
        return LeastConfident(args)
    elif args.method == 'coreset':
        return Coreset(args)
    elif args.method == 'badge':
        return BADGE(args)
    elif args.method == 'bald':
        return Bald(args)
    elif args.method == 'uwe':
        return UWE(args)
    else:
        raise NotImplementedError
        

def get_model(args):
    n_classes = const.dataset[args.dataset]['n_classes']
    n_channels = const.dataset[args.dataset]['n_channels']

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        cifar_stem = True
    else:
        cifar_stem = False

    if args.model == 'resnet18':
        return resnet.ResNet18(n_channels=n_channels, n_classes=n_classes, cifar_stem=cifar_stem)
    elif args.model == 'resnet34':
        return resnet.ResNet34(n_channels=n_channels, n_classes=n_classes, cifar_stem=cifar_stem)
    elif args.model == 'resnet50':
        return resnet.ResNet50(n_channels=n_channels, n_classes=n_classes, cifar_stem=cifar_stem)
    elif args.model == 'resnet101':
        return resnet.ResNet101(n_channels=n_channels, n_classes=n_classes, cifar_stem=cifar_stem)
    elif args.model == 'resnet152':
        return resnet.ResNet152(n_channels=n_channels, n_classes=n_classes, cifar_stem=cifar_stem)
    elif args.model == 'vit':
        return vision_transformer.VisionTransformerClassifier(n_label=n_classes, fine_tune_layers=args.fine_tune_layers)
    else:
        raise NotImplementedError


def get_run_id(args, action=None):
    if action is None:
        action = args.action

    run_id = f'{args.dataset}_{action}_mod{args.model}_s{args.seed}_ne{args.n_epoch}'

    if action == 'al':
        run_id += '_bid' + args.budget_id
        run_id += '_m' + args.method
        if len(args.method_tag) > 0:
            run_id += args.method_tag
    elif action == 'alc0':
        run_id += '_bid' + args.budget_id
    elif action == 'initialize':
        run_id = args.dataset + '_bid' + args.budget_id
    
    if (action == 'al' or action == 'supervised') and len(args.run_tag) > 0:
        run_id += '_ct' + args.run_tag

    return run_id


def get_result_folder(args, action=None):
    if action is None:
        action = args.action
    if action == 'al':
        folder = os.path.join('results', args.dataset, args.method + args.method_tag)
    elif action == 'alc0' or action == 'initialize':
        folder = os.path.join('results', args.dataset, 'initialize')
    elif action == 'supervised':
        folder = os.path.join('results', args.dataset, 'supervised')
    else:
        raise NotImplementedError
    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder


def get_init_folder(args):
    return os.path.join('results', args.dataset, 'initialize')


def get_init_file(args, type='init_lb'):
    return args.dataset + '_bid' + args.budget_id + '_' + type + '.csv'


def get_result_files(args):
    files = const.al_result_files[args.dataset][args.budget_id]
    return files
