import time

import numpy as np
import torch

from .strategy import Strategy


class Bvsb(Strategy):
    def __init__(self, args):
        super(Bvsb, self).__init__(args)
        self.args.uncertainty = 'bvsb'

    def query(self, n):
        unlabeled_data = self.get_unlabeled_data()
        unlabeled_idxs = unlabeled_data['idxs']

        unlb_result = self.test('tr', idx=unlabeled_idxs)
        entropy = self.get_uncertainty(unlb_result)

        st = time.time()
        chosen = unlabeled_idxs[torch.argsort(entropy, descending=True)[:n]]
        et = time.time()
        self.query_time.append(et - st)

        return chosen
