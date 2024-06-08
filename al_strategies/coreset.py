import time

from .strategy import Strategy

class Coreset(Strategy):
    def __init__(self, args):
        super(Coreset, self).__init__(args)
        self.args.selection = 'k_center_greedy'

    def query(self, n):
        unlabeled_data = self.get_unlabeled_data()
        unlabeled_idxs = unlabeled_data['idxs']
        test_result = self.test('tr')
        feats = test_result['features']

        st = time.time()
        chosen = self.select(feats.numpy(), n)
        et = time.time()
        self.query_time.append(et - st)

        chosen = unlabeled_idxs[chosen]
        
        return chosen
