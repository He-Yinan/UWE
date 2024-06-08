import time

import numpy as np

from .strategy import Strategy


class Random(Strategy):
    def __init__(self, args):
        super(Random, self).__init__(args)

    def query(self, n):
        st = time.time()
        chosen = np.random.choice(np.where(self.lb_flag == 0)[0], n, replace=False)
        et = time.time()

        self.query_time.append(et - st)

        return chosen
