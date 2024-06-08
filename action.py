import os
import numpy as np

from utils import logger, const, read_write, getter
from al_strategies import strategy


def initialize(args):
    l = logger.Logger.get(args)  
    folder = getter.get_init_folder(args)
    lb_idxs_file = getter.get_init_file(args, 'init_lb')
    vd_idxs_file = getter.get_init_file(args, 'vd')
    n_init_lb = const.budget[args.budget_id]['lb'][0]
    n_vd = const.budget[args.budget_id]['vd']
    tr_size = const.dataset[args.dataset]['tr']

    idxs = np.arange(tr_size)
    np.random.shuffle(idxs)

    lb_idxs = idxs[0:n_init_lb]
    vd_idxs = idxs[n_init_lb:(n_init_lb + n_vd)]

    n_duplicates = len(np.intersect1d(lb_idxs, vd_idxs))
    assert n_duplicates == 0

    l.info(f'n initial labeled: {len(lb_idxs)}, n validation: {len(vd_idxs)}, check duplicates: {n_duplicates}')

    read_write.write_csv(['lb_idxs'], folder, lb_idxs_file, mode='w')
    for i in lb_idxs:
        read_write.write_csv([i], folder, lb_idxs_file, mode='a')
    read_write.write_csv(['vd_idxs'], folder, vd_idxs_file, mode='w')
    for i in vd_idxs:
        read_write.write_csv([i], folder, vd_idxs_file, mode='a')


def active_learning_r0(args):
    strat = strategy.Strategy(args)
    strat.train()
    _ = strat.test('te')


def supervised(args):
    strat = strategy.Strategy(args)
    strat.train()
    _ = strat.test('te')


def active_learning(args):
    l = logger.Logger.get(args)
    budgets = const.budget[args.budget_id]['lb']
    folder = getter.get_result_folder(args)
    al_cycle_result_file = args.run_id + '_al_cycle_result.csv'
    read_write.write_csv(['n_labeled', 'te_accuracy', 'q_accuracy', 'query_time', 'tr_loss', 'lb_loss', 'unlb_loss', 'q_loss'], 
                         os.path.join(folder, 'al_cycle_result'),
                         al_cycle_result_file, mode='w')

    al_cycle_result_info = {}
    al_cycle_result_info['te_accuracy'] = []
    al_cycle_result_info['q_accuracy'] = []
    al_cycle_result_info['query_time'] = [0]

    # al cycle 0
    strat = getter.get_strategy(args)
    model_folder = os.path.join(getter.get_result_folder(args, action='alc0'), 'train')
    model_file = getter.get_run_id(args, action='alc0') + '_cycle' + str(strat.al_cycle) + '_model.pt'
    lb_idxs = strat.get_labeled_data()['idxs']
    unlabeled_idxs = strat.get_unlabeled_data()['idxs']
    tr_result = strat.test('tr')
    lb_result = strat.test('tr', idx=lb_idxs)
    unlb_result = strat.test('tr', idx=unlabeled_idxs)
    al_cycle_result_info['q_accuracy'].append(tr_result['accuracy'])

    strat.load_model(os.path.join(model_folder, model_file))
    
    te_result = strat.test('te')
    al_cycle_result_info['te_accuracy'].append(te_result['accuracy'])
    read_write.write_csv([budgets[0], al_cycle_result_info['te_accuracy'][-1], al_cycle_result_info['q_accuracy'][-1], al_cycle_result_info['query_time'][-1], tr_result['loss'], lb_result['loss'], unlb_result['loss'], tr_result['loss']],
                         os.path.join(folder, 'al_cycle_result'),
                         al_cycle_result_file, mode='a')

    # al cycle
    for al_cycle, _ in enumerate(budgets[:-1], start=1):
        strat.al_cycle = al_cycle
        n_query = budgets[al_cycle] - budgets[al_cycle-1]
        l.info('')
        l.info(f'running al cycle {al_cycle}, query {n_query}, current labeled: {budgets[al_cycle-1]}')

        # query
        q_idxs = strat.query(n_query)
        strat.update(q_idxs)
        assert budgets[al_cycle] == strat.get_labeled_data()['count']

        # al tests
        lb_idxs = strat.get_labeled_data()['idxs']
        unlabeled_idxs = strat.get_unlabeled_data()['idxs']
        tr_result = strat.test('tr')
        uncert_tr_result = tr_result
        q_result = strat.test('tr', idx=q_idxs)
        lb_result = strat.test('tr', idx=lb_idxs)
        unlb_result = strat.test('tr', idx=unlabeled_idxs)

        # get al cycle query results
        q_labels = strat.get_data(q_idxs)['y']
        q_acc = q_result['accuracy']
        q_preds = q_result['predictions']
        al_cycle_result_info['q_accuracy'].append(q_acc)
        al_cycle_result_info['query_time'].append(strat.query_time[-1])

        # get al cycle uncertainty, feature
        tr_feats = tr_result['features']
        tr_uncertainty = strat.get_uncertainty(uncert_tr_result)

        # train
        strat.train()

        # get al cycle test results
        te_acc = strat.test('te')['accuracy']
        al_cycle_result_info['te_accuracy'].append(te_acc)

        # save al cycle results
        read_write.write_csv([strat.get_labeled_data()['count'], al_cycle_result_info['te_accuracy'][-1], al_cycle_result_info['q_accuracy'][-1], al_cycle_result_info['query_time'][-1], tr_result['loss'], lb_result['loss'], unlb_result['loss'], q_result['loss']],
                             os.path.join(folder, 'al_cycle_result'),
                             al_cycle_result_file, mode='a')


        l.info(f'saved al cycle result info to {al_cycle_result_file}')
        
        # save al query results
        q_file = args.run_id + f'_cycle{str(al_cycle)}_query.csv'
        read_write.write_csv(['rank', 'idx', 'label', 'pred'], 
                             os.path.join(folder, 'query'),
                             q_file, mode='w')
        for i in range(n_query):
            read_write.write_csv([i, q_idxs[i], q_labels[i].item(), q_preds[i].item()],
                                 os.path.join(folder, 'query'),
                                 q_file, mode='a')
        l.info(f'saved query info to {q_file}')
        
        # save uncertainty, feature
        tr_uncertainty_file = args.run_id + f'_cycle{str(al_cycle)}_tr{args.uncertainty}.pt'
        tr_feat_file = args.run_id + f'_cycle{str(al_cycle)}_trfeat.pt'
        read_write.write_pt(tr_uncertainty, os.path.join(folder, 'query'), tr_uncertainty_file)
        read_write.write_pt(tr_feats, os.path.join(folder, 'query'), tr_feat_file)
        l.info(f'saved uncertainties to {tr_uncertainty_file}, features to {tr_feat_file}')

        l.info(f'========== al cycle accuracy: {al_cycle}, {te_acc} ==========')
