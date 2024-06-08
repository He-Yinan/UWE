import os
import time
from pathlib import Path

from .strategy import Strategy
from utils import read_write, getter
import numpy as np


class BADGE(Strategy):
    def __init__(self, args):
        super(BADGE, self).__init__(args)
        self.args.selection = 'kmeans_pp'

    def query(self, n):
        unlabeled_data = self.get_unlabeled_data()
        unlabeled_idxs = unlabeled_data['idxs']
        unlb_x = unlabeled_data['x']
        unlb_result = self.test('tr', idx=unlabeled_idxs)
        unlb_pred = unlb_result['predictions']

        data = self.get_tr_data()
        tr_x = data['x']
        tr_result = self.test('tr')
        tr_pred = tr_result['predictions']

        st = time.time()
        grad_embedding = self.get_badge_gradient_embedding(unlb_x, unlb_pred)
        et = time.time()
        self.l.info(f'BADGE feature extraction time: {et - st}')

        feat_folder = os.path.join(getter.get_result_folder(self.args), 'feature')
        Path(feat_folder).mkdir(parents=True, exist_ok=True)
        read_write.write_pt(grad_embedding, feat_folder, f'grad_emb_{self.args.run_id}_r{str(self.al_cycle)}.pt')
        
        st = time.time()
        chosen = self.select(grad_embedding.numpy().astype('float32'), n)
        et = time.time()
        self.query_time.append(et-st)

        np.save(os.path.join(feat_folder, f'chosen_local_idx_s{self.args.seed}_r{self.al_cycle}_rt{self.args.run_tag}.npy'), chosen)

        chosen = unlabeled_idxs[chosen]

        return chosen
