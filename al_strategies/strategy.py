import os
import pdb
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from sklearn.metrics import pairwise_distances
from scipy import stats

import dataset
from utils import logger, getter, const, read_write

if torch.cuda.is_available():
    from utils.cuda.feat_dist_gpu import feat_dist_gpu


class Strategy(object):

    def __init__(self, args):
        self.args = args
        self.l = logger.Logger.get(args)
        self.al_cycle = self.args.al_cycle
        self.query_time = []
        
        data = dataset.get_dataset(args)
        self.handler = dataset.get_handler(self.args.dataset)
        self.x_tr, self.x_vd, self.x_te = data['x_tr'], data['x_vd'], data['x_te']
        self.y_tr, self.y_vd, self.y_te = data['y_tr'], data['y_vd'], data['y_te']
        self.lb_flag = data['lb_flag']

        # gpu
        use_cuda = torch.cuda.is_available()
        self.device = torch.device(
            f'cuda:{self.args.gpu_id}' if use_cuda else 'cpu')
        self.l.info(f'gpu device:{self.device}')

        # dataset information
        self.args.dataset_info = const.dataset[self.args.dataset]

        # model
        self.net = getter.get_model(self.args)
        self.clf = self.net.to(self.device)
 
    def query(self, n):
        pass 

    def update(self, q_idxs):
        old_lb_count = self.get_labeled_data()['count']
        self.l.info(f'old labeled indexes count: {old_lb_count}')
        self.l.info(f'new labeled indexes count: {len(q_idxs)}')
        self.lb_flag[q_idxs] = True 
        total_lb_count = self.get_labeled_data()['count']
        self.l.info(f'total labeled indexes count: {total_lb_count}')

    def get_labeled_data(self):
        lb_idx = np.arange(len(self.y_tr))[self.lb_flag]
        return {
            'x': self.x_tr[lb_idx],
            'y': self.y_tr[lb_idx],
            'count': len(lb_idx),
            'idxs': lb_idx
        }

    def get_unlabeled_data(self):
        unlb_idx = np.arange(len(self.y_tr))[~self.lb_flag]
        return {
            'x': self.x_tr[unlb_idx],
            'y': self.y_tr[unlb_idx],
            'count': len(unlb_idx),
            'idxs': unlb_idx
        }

    def get_data(self, idxs):
        return {
            'x': self.x_tr[idxs],
            'y': self.y_tr[idxs],
            'count': len(idxs)
        }

    def get_tr_data(self):
        return {
            'x': self.x_tr,
            'y': self.y_tr,
            'count': len(self.y_tr)
        }

    def _train(self, loader_tr, opt, n_samples):
        self.clf.train()
        epoch_loss = 0.0
        epoch_correct = 0.0
        
        for _, (x, y, _) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            batch_size = len(y)

            opt.zero_grad()
            out, _ = self.clf(x)
            _, preds = torch.max(out, 1)
            loss = F.cross_entropy(out, y)  # cross entropy between logits and targets
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * batch_size
            epoch_correct += float(torch.sum(preds == y))
        
        epoch_loss = epoch_loss / n_samples
        epoch_acc = epoch_correct / n_samples
        return {
            'epoch_loss': epoch_loss,
            'epoch_tr_acc': epoch_acc
        }

    def train(self, model_path=None):
        if model_path is not None:
            self.load_model(model_path)
        else:
            self.net = getter.get_model(self.args)
            self.clf = self.net.to(self.device)
        
        if self.args.model == 'vit':
            opt = optim.Adam(self.clf.parameters(), lr=5e-5)
        else:
            opt = optim.SGD(self.clf.parameters(), lr=0.02, momentum=0.9, nesterov=True, weight_decay=5e-4)
            scheduler = MultiStepLR(opt, milestones=[60, 80], gamma=0.5)
        
        tr_idxs = np.arange(len(self.y_tr))[self.lb_flag]
        self.l.info(f'train using {len(tr_idxs)} samples')

        if len(tr_idxs) == 0:
            end_epoch = 1
            epoch = self.args.n_epoch
        else:
            end_epoch = self.args.n_epoch + 1
            loader_tr = DataLoader(self.handler(self.x_tr[tr_idxs], self.y_tr[tr_idxs], transform=self.args.dataset_info['transform']),
                                shuffle=True, **self.args.dataset_info['loader_tr_args'])

        self.l.info(f'initialized loader_tr')

        tr_info = {}
        tr_info['epoch'] = []
        tr_info['tr_loss'] = []
        tr_info['tr_acc'] = []
        tr_info['te_loss'] = []
        tr_info['te_acc'] = []

        for epoch in range(1, end_epoch):
            epoch_info = self._train(loader_tr, opt, len(tr_idxs))
            if self.args.model != 'vit':
                scheduler.step(epoch)

            # save epoch info
            epoch_loss = epoch_info['epoch_loss']
            epoch_tr_acc = epoch_info['epoch_tr_acc']
            if self.args.full_train == 1:
                te_info = self.test('te')
                te_loss = te_info['loss']
                te_acc = te_info['accuracy']
            else:
                te_loss = 0
                te_acc = 0

            self.l.info(f'al cycle: {self.al_cycle}, epoch: {epoch}, tr_loss: {epoch_loss}, tr_acc: {epoch_tr_acc}, te_acc: {te_acc}, te_loss: {te_loss}' )
            tr_info['epoch'].append(epoch)
            tr_info['tr_loss'].append(epoch_loss)
            tr_info['tr_acc'].append(epoch_tr_acc)
            tr_info['te_loss'].append(te_loss)
            tr_info['te_acc'].append(te_acc)
        
        self.l.info(tr_info)

        # save training info
        folder = os.path.join(getter.get_result_folder(self.args), 'train')
        file = self.args.run_id + '_cycle' + str(self.al_cycle) + '_tr_info.csv'
        read_write.write_csv(['epoch', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc'], folder, file, mode='w')
        if len(tr_idxs) > 0:
            for i in range(self.args.n_epoch):
                read_write.write_csv([tr_info['epoch'][i], tr_info['tr_loss'][i], tr_info['tr_acc'][i], tr_info['te_loss'][i], tr_info['te_acc'][i]], folder, file, mode='a')

        # save model
        self.save_model(epoch, opt)

    def test(self, split, idx=None, model_path=None):
        if model_path is not None:
            self.load_model(model_path)

        if split == 'tr':
            X, Y = self.x_tr, self.y_tr
        elif split == 'vd':
            X, Y = self.x_vd, self.y_vd
        elif split == 'te':
            X, Y = self.x_te, self.y_te
        if idx is not None:
            X, Y = X[idx], Y[idx]
        
        normalize = self.args.dataset_info['normalize']
        loader_te = DataLoader(self.handler(X, Y, transform=transforms.Compose([transforms.ToTensor(), normalize])), **self.args.dataset_info['loader_te_args'])

        features = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        logits = torch.zeros([len(Y), self.args.dataset_info['n_classes']])
        probabilities = torch.zeros([len(Y), self.args.dataset_info['n_classes']])
        predictions = torch.zeros(len(Y), dtype=Y.dtype)
        losses = torch.zeros(len(Y), dtype=torch.float)
        total_loss = 0.0

        self.clf.eval()
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                logit, feature = self.clf(x)  # resnet
                _loss = F.cross_entropy(logit, y, reduction='none')
                probability = F.softmax(logit, dim=1)
                prediction = logit.max(1)[1]
                
                features[idxs] = feature.cpu()
                logits[idxs] = logit.cpu()
                losses[idxs] = _loss.cpu()
                probabilities[idxs] = probability.cpu()
                predictions[idxs] = prediction.cpu()
                total_loss += _loss.cpu().sum().item()
        
        if len(loader_te) > 0:
            accuracy = 1.000 * (Y.cpu() == predictions).sum().item() / len(Y.cpu())
            loss = (total_loss / len(Y.cpu()))
        else:
            accuracy = -1
            loss = -1
        
        self.l.info(f'accuracy: {accuracy}, features: {features.shape}, logits: {logits.shape}, probabilities: {probabilities.shape}, predictions: {predictions.shape}, loss: {loss}')

        return {
            'accuracy': accuracy,
            'features': features,
            'logits': logits,
            'probabilities': probabilities,
            'predictions': predictions,
            'loss': loss,
            'losses': losses
        }
    
    def get_badge_gradient_embedding(self, X, Y):
        normalize = self.args.dataset_info['normalize']
        loader_te = DataLoader(self.handler(X, Y, transform=transforms.Compose([transforms.ToTensor(), normalize])), **self.args.dataset_info['loader_tr_args'])

        emb_dim = self.clf.get_embedding_dim()
        n_lab = len(np.unique(Y))
        embedding = np.zeros([len(Y), emb_dim * n_lab])

        self.clf.eval()
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                logit, feature = self.clf(x)  # resnet
                feature = feature.data.cpu().numpy()
                probability = F.softmax(logit, dim=1).data.cpu().numpy()
                prediction = np.argmax(probability, 1)

                for j in range(len(y)):
                    for c in range(n_lab):
                        if c == prediction[j]:
                            embedding[idxs[j]][emb_dim * c: emb_dim * (c + 1)] = deepcopy(feature[j]) * (1 - probability[j][c])
                        else:
                            embedding[idxs[j]][emb_dim * c: emb_dim * (c + 1)] = deepcopy(feature[j]) * (-1 * probability[j][c])
        
        return torch.Tensor(embedding)

    def get_n_predictions(self, x, y):
        normalize = self.args.dataset_info['normalize']
        loader_te = DataLoader(self.handler(x, y, transform=transforms.Compose([transforms.ToTensor(), normalize])), **self.args.dataset_info['loader_tr_args'])

        self.clf.train()  # apply dropout
        probs = torch.zeros([self.args.n_pass, len(y), len(np.unique(y))])
        for i in range(self.args.n_pass):
            self.l.info(f'n_pass {i + 1}/{self.args.n_pass}')
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    logit, _ = self.clf(x)  # resnet
                    probs[i][idxs] += F.softmax(logit, dim=1).cpu()
        return probs
    
    def save_model(self, epoch, opt, path=None):
        if path is None:
            if self.al_cycle == 0 and self.args.action == 'al':
                folder = os.path.join(getter.get_init_folder(self.args), 'train')
            else:
                folder = os.path.join(getter.get_result_folder(self.args), 'train')
            file = self.args.run_id + '_cycle' + str(self.al_cycle) + '_model.pt'
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, file)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.clf.state_dict(),
            'optimizer_state_dict': opt.state_dict()},
            path)

        self.l.info(f'saved model {path}')
    
    def load_model(self, path):
        '''
        path: model path
        '''
        self.net = getter.get_model(self.args)
        self.clf = self.net.to(self.device)
        model = torch.load(path, map_location='cpu')
        self.clf.load_state_dict(model['model_state_dict'])
        self.l.info(f'loaded model from {path}')

    def cross_entropy(self, predictions, targets, axis, epsilon=1e-12):

        """Computes cross entropy between targets (encoded as one-hot vectors) and predictions.
        
        Args: 
            predictions: (N, k) ndarray
            targets: (N, k) ndarray  

        Return: 
            float
        """

        predictions = np.clip(predictions, epsilon, 1.-epsilon)
        ce = - np.sum(targets * np.log(predictions + 1e-9), axis=axis)
        return ce
    
    def get_uncertainty(self, te_result):
        if self.args.uncertainty == 'entropy':
            probs = te_result['probabilities']
            entropy = (-probs * torch.log(probs)).sum(1) 
            return entropy
        elif self.args.uncertainty == 'bvsb':
            probs = te_result['probabilities']
            self.l.info(f'probs shape: {probs.shape}')
            sorted_probs = probs.sort()[0]
            b_probs = sorted_probs[:,-1]
            sb_probs = sorted_probs[:,-2]
            bvsb = sb_probs / b_probs
            self.l.info(f'bvsb shape: {bvsb.shape}')
            return bvsb
        elif self.args.uncertainty == 'least_confident':
            probs = te_result['probabilities']
            self.l.info(f'probs shape: {probs.shape}')
            sorted_probs = probs.sort()[0]
            b_probs = sorted_probs[:,-1]
            self.l.info(f'b_probs shape: {b_probs.shape}')
            return 1 - b_probs
        else:
            self.l.error(f'uncertainty type {self.args.uncertainty} is not implemented')
            raise NotImplementedError

    def select(self, x, k):
        '''
        x: embedding (numpy)
        k: number of samples to select
        '''
        selected = None
        self.l.info(f'using {self.args.selection} selection')
        if self.args.selection == 'kmeans_pp':
            selected = self.kmeans_pp(x, k)
        elif self.args.selection == 'k_center_greedy_unlb':
            selected = self.k_center_greedy_unlb(x, k)
        elif self.args.selection == 'k_center_greedy':
            selected = self.k_center_greedy(x, k)
        return selected
    
    def kmeans_pp(self, x, k):
        self.l.info('running kmeans++')
        ind = np.argmax([np.linalg.norm(s, 2) for s in x])
        self.l.info(f'first selected idx: {ind}')
        mu = [x[ind]]
        indsAll = [ind]
        centInds = [0.] * len(x)
        cent = 0

        while len(mu) < k:
            if len(mu) % 10 == 0:
                self.l.info(f'mu length: {len(mu)}, k: {k}, cent: {cent}')

            if len(mu) == 1:
                D2 = pairwise_distances(x, mu).ravel().astype('float32')
            else:
                newD = feat_dist_gpu(x, mu[-1].astype('float32'), 0, self.args.gpu_id)            
                newD = newD.reshape(-1, 1)
                self.l.info(f'calculated new distance to selected data')
                for i in range(len(x)):
                    if i == ind:
                        centInds[i] = cent
                        D2[i] = 0
                    elif D2[i] > newD[i]:
                        D2[i] = newD[i]

            self.l.info(f'sum D2: {str(sum(D2))}, min D2: {str(D2.min())}, max D2: {str(D2.max())}')

            if sum(D2) == 0.0:
                pdb.set_trace()
                self.l.error('sum D2 is zero')
                raise ValueError

            D2 = D2.ravel().astype(np.float64)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            self.l.info(f'sum Ddist: {str(sum(Ddist))}, min Ddist: {str(Ddist.min())}, max Ddist: {str(Ddist.max())}')

            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            self.l.info(f'selected index Ddist: {Ddist[ind]}, distance: {D2[ind]}')

            loop_count = 0
            while ind in indsAll:
                loop_count += 1
                ind = customDist.rvs(size=1)[0]
            self.l.info(f'exited kmeans++ while loop, loop count: {loop_count}, selected index: {ind}')
            
            mu.append(x[ind])
            indsAll.append(ind)
            cent += 1

        self.l.info(f'kmeans++ selected {len(indsAll)} samples')
        return indsAll

    def k_center_greedy_unlb(self, x, k):
        self.l.info('running k center greedy (unlb)')
        selected_idxs = np.zeros(k, dtype=int)
        init_selected = np.argmax([np.linalg.norm(feat, 2) for feat in x])
        selected_idxs[0] = init_selected
        dist = feat_dist_gpu(x.astype('float32'), x[init_selected].astype('float32'), 0, self.args.gpu_id)
        dist[init_selected] = 0
        self.l.info(f'dist shape: {dist.shape}')

        for i in range(1, k):
            if i % 10 == 0:
                self.l.info(f'selecting {i}')
            selected_idx = np.argmax(dist)
            selected_idxs[i] = selected_idx
            new_dist = feat_dist_gpu(x.astype('float32'), x[selected_idx].astype('float32'), 0, self.args.gpu_id)
            dist = np.minimum(dist, new_dist)
            dist[selected_idx] = 0
        return selected_idxs

    def k_center_greedy(self, x, k):
        '''
        x: embedding of all training samples (numpy array)
        '''
        self.l.info('running k center greedy')

        feats_lb = x[self.lb_flag].astype('float32')
        feats_unlb = x[~self.lb_flag].astype('float32')

        dists = []
        for i in range(feats_unlb.shape[0]): # as features are updated for each batch, feature distances have to be re-computed
            if i % 1000 == 0:
                self.l.info(f'initializing distance: {i}/{feats_unlb.shape[0]}')
            dist = feat_dist_gpu(feats_lb, feats_unlb[i], metric=0)  
            dists.append(np.min(dist))
        dists = np.array(dists).reshape(-1, 1)
        self.l.info(f'k center greedy initialized distance ({dists.shape}) between {feats_unlb.shape[0]} unlabeled features and {feats_lb.shape[0]} labeled features')

        mu = []
        inds_all = []
        while len(mu) < k:
            if sum(dists) == 0.0: 
                raise Exception('sum dists is zero')
            
            if len(mu) % 10 == 0:
                self.l.info(f'selecting {len(mu)}')
            
            ind = np.argmax(dists)
            mu.append(feats_unlb[ind])
            inds_all.append(ind)
            
            newD = feat_dist_gpu(feats_unlb, mu[-1], metric=0).reshape(-1, 1)
            dists = np.minimum(dists, newD)
        
        return inds_all
