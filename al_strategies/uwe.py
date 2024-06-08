import os
import time

from .strategy import Strategy
from utils import read_write, getter

class UWE(Strategy):
    def __init__(self, args):
        super(UWE, self).__init__(args)
        self.args.selection = 'k_center_greedy_unlb'
        if self.args.uncertainty == 'none':
            raise ValueError

    def query(self, n):
        unlabeled_data = self.get_unlabeled_data()
        unlabeled_idxs = unlabeled_data['idxs']
        folder = getter.get_result_folder(self.args)
        read_write.write_npy(unlabeled_idxs, os.path.join(folder, 'query'), self.args.run_id + f'_cycle{str(self.al_cycle)}_uunlabeled_idxs.npy')

        st = time.time()
        test_result = self.test('tr')

        # get features
        feats = test_result['features']
        feats = feats[unlabeled_idxs]

        # get uncertainties
        uncertainty = self.get_uncertainty(test_result)
        uncertainty = uncertainty[unlabeled_idxs]
        uncertainty_norm = ((uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())).unsqueeze(1)
        self.l.info(f'uncertainty_norm shape: {uncertainty_norm.shape}')

        # uncertainty weighted embeddings
        uwe = uncertainty_norm * feats
        self.l.info(f'uwe shape: {uwe.shape}')
        et = time.time()
        self.l.info(f'uwe feature extraction time: {et - st}')

        # selection
        st = time.time()
        chosen = self.select(uwe.numpy(), n)
        et = time.time()
        self.query_time.append(et - st)
        self.l.info(f'uwe selection time: {et - st}')

        global_chosen = unlabeled_idxs[chosen]
        
        return global_chosen
