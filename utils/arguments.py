import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--action', type=str, default='al', choices=['initialize', 'alc0', 'al', 'supervised'])
    parser.add_argument('--method', type=str, default='uwe', choices=['uwe', 'random', 'entropy', 'bvsb', 'least_confident', 'coreset', 'badge', 'bald'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mini_imagenet'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--budget_id', type=str, required=True)
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vit'])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--n_epoch', type=int, required=True)
    parser.add_argument('--al_cycle', type=int, default=0)
    parser.add_argument('--method_tag', type=str, default='')
    parser.add_argument('--run_tag', type=str, default='')

    parser.add_argument('--full_train', type=int, default=0)  # save loss and accuracy of test set during training
    parser.add_argument('--uncertainty', type=str, default='entropy', choices=['entropy', 'bvsb', 'least_confident'])
    parser.add_argument('--selection', type=str, default='none', choices=['kmeans_pp', 'k_center_greedy', 'k_center_greedy_unlb', 'none'])
    parser.add_argument('--n_pass', type=int, default=5)
    parser.add_argument('--fine_tune_layers', type=int, default=1)
    
    return parser.parse_args()
