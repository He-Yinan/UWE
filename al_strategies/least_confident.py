import time

import torch

from .strategy import Strategy


class LeastConfident(Strategy):
    def __init__(self, args):
        super(LeastConfident, self).__init__(args)
        self.args.uncertainty = 'least_confident'

    def query(self, n):
        unlabeled_data = self.get_unlabeled_data()
        unlabeled_idxs = unlabeled_data['idxs']

        unlb_result = self.test('tr', idx=unlabeled_idxs)
        not_conf = self.get_uncertainty(unlb_result)

        st = time.time()
        chosen = unlabeled_idxs[torch.argsort(not_conf, descending=True)[:n]]
        et = time.time()
        self.query_time.append(et-st)

        return chosen
