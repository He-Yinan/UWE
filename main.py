import numpy as np
import torch

import action
import dataset
import utils.arguments as arguments
from utils import logger
from utils import getter


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True

    l = logger.Logger.get(args)
    args.run_id = getter.get_run_id(args)
    l.info(f'arguments: {args}')

    if args.action == 'initialize':
        action.initialize(args)
    elif args.action =='alc0':
        action.active_learning_r0(args)
    elif args.action == 'al':
        action.active_learning(args)
    elif args.action == 'supervised':
        action.supervised(args)
    else:
        l.error(f'{args.action} is not implemented')
        raise NotImplementedError


if __name__ == '__main__':
    args = arguments.get_args()
    main(args)
