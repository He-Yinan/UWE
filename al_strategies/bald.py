import time

import torch

from .strategy import Strategy


class Bald(Strategy):
    def __init__(self, args):
        super(Bald, self).__init__(args)

    def query(self, n):
        unlabeled_data = self.get_unlabeled_data()
        unlabeled_idxs = unlabeled_data['idxs']
        unlb_x = unlabeled_data['x']
        unlb_y = unlabeled_data['y']

        st = time.time()
        probs = self.get_n_predictions(unlb_x, unlb_y)
        et = time.time()
        self.query_time.append(et - st)

        mean_probs = probs.mean(0)
        entropy1 = (- mean_probs * torch.log(mean_probs)).sum(1)
        entropy2 = (- probs * torch.log(probs)).sum(2).mean(0)
        U = entropy2 - entropy1
        chosen = unlabeled_idxs[U.sort()[1][:n]]

        return chosen
