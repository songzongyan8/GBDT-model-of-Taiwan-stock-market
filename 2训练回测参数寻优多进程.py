import os
import time
import pickle
import numpy as np
import pandas as pd
from train import ProData, TrainModel
from multiprocessing import Pool
from memory_profiler import profile


def GetParas(features):
    categorical_feature = []
    dt_params = {'max_depth': 7,
                 'min_data_in_leaf': 20,
                 'min_child_weight': 1e-8,
                 'feature_fraction': 0.8,
                 'feature_fraction_seed': 2,
                 'sub_row': 0.9,
                 'bagging_freq': 5,
                 'lambda_l1': 0.35704560163196114,
                 'lambda_l2': 0.04559386749660369,
                 'min_gain_to_split': 0.9059580665888065,
                 'min_data_per_group': 10,
                 'max_cat_threshold': 32,
                 'cat_smooth': 10,
                 'cat_l2': 10,
                 'max_cat_to_onehot': 4,
                 # 'max_cat_group': 10,
                 # 'Task': 'train', # train, predict, convert_model
                 'boosting': 'gbdt',  # gbdt, rf, dart, goss
                 'num_leaves': 255,  # 31
                 'max_bin': 255,
                 'min_data_in_bin': 3,
                 'top_k': 3,
                 'boost_from_average': False,
                 'sample_weight': None,
                 'learning_rate': 0.22400795868824444,
                 'objective': 'binary',
                 'metric': ['binary_logloss', 'auc'],
                 # 'num_classes': 1,
                 'train_metric': True,
                 # 'tree_learner': 'serial',  # serial, feature, data, voting
                 'num_threads': 1,  # 'OpenMP_default'
                 'device': 'cpu',
                 # 'weight': 'zf',
                 'verbose': -1,
                 }

    # 参数遍历
    para_list = []
    for num_boost_round in [5000]:
        for early_stopping_rounds in [50]:  # [100]:
            for max_depth in [pow(2, i) - 1 for i in range(7, 21)]:  # choice [pow(2, i) - 1 for i in range(2, 11)]:  #
                for min_gain_to_split in [0.1 * i for i in range(1, 11)]:  # uniform [0.01 * i for i in range(1, 101)]:  #
                    for top_k in [21]:  # choice  [20 * i + 1 for i in range(1, 101, 5)]:  #
                        for num_leaves in [2 ** i - 1 for i in range(3, 17)]:  # choice [2 ** i - 1 for i in range(3, 17)]:  #
                            for learning_rate in [0.005]:  # uniform [0.005 * i for i in range(1, 101)]:  #
                                for lambda_l1 in [0.5]:  # [0.256]:  # uniform  [0.008 * i for i in range(1, 101)]:  #
                                    for lambda_l2 in [0.5]:  # uniform  [0.008 * i for i in range(1, 101)]:  #
                                        for feature_fraction in [i / 7 for i in range(3, 8)]:  # choice [i / 7 for i in range(3, 8)]:  #
                                            for sub_row in [0.8]:  # choice [i * 0.1 for i in range(3, 11)]:  #
                                                if num_leaves >= pow(2, max_depth):
                                                    continue
                                                update_dict = {'max_depth': max_depth,
                                                               'min_gain_to_split': min_gain_to_split,
                                                               'top_k': top_k,
                                                               'num_leaves': num_leaves,
                                                               'learning_rate': learning_rate,
                                                               'lambda_l1': lambda_l1,
                                                               'lambda_l2': lambda_l2,
                                                               'feature_fraction': feature_fraction,
                                                               'sub_row': sub_row}
                                                dt_params.update(update_dict)
                                                para = {'features': features.copy(),
                                                        'categorical_feature': categorical_feature.copy(),
                                                        'num_boost_round': num_boost_round,
                                                        'early_stopping_rounds': early_stopping_rounds,
                                                        'dt_params': dt_params.copy()}
                                                para_list.append(para.copy())
    return(para_list)


def main(data_path, label_column,
         train_start_time, val_start_time, test_start_time,
         res_path, para, idx):

    sss = time.perf_counter()
    # 设定参数
    features = para['features']
    dt_params = para['dt_params']
    categorical_feature = para['categorical_feature']
    num_boost_round = para['num_boost_round']
    early_stopping_rounds = para['early_stopping_rounds']

    # 训练
    df_val, df_test, yval, ytest, pred_val, pred_test, metrics = TrainModel(data_path, features, label_column,
                                                                            train_start_time, val_start_time, test_start_time,
                                                                            dt_params, categorical_feature, num_boost_round, early_stopping_rounds)

    ''' 保存txt '''
    with open(res_path, 'at') as f1:
        t0 = 'features: ' + str(features)
        t1 = 'dt_params: ' + str(dt_params)
        t2 = 'categorical_feature: ' + str(categorical_feature)
        t3 = 'num_boost_round: ' + str(num_boost_round)
        t4 = 'early_stop_rounds: ' + str(early_stopping_rounds)
        t5 = 'trn_acc: {:.3f}\nval_acc: {:.3f}\ntest_acc: {:.3f}'.format(*metrics[:3])
        t6 = 'trn_report: \n{}\nval_report: \n{}\ntest_report: \n{}'.format(*metrics[3:])
        text = ('\n').join([t0, t1, t2, t3, t4, t5, t6])
        f1.write(text + '\n\n')
    print('本轮优化结束-{}！！！！！！！！！！！！！！！！！！'.format(idx), time.perf_counter() - sss)


if __name__ == '__main__':
    # 设置参数
    data_path = './processed_data.csv'
    feature_cols = ['負債總計', '營業活動之淨現金流入（流出）',
                    '投資活動之淨現金流入（流出）', '籌資活動之淨現金流入（流出）', '營運產生之現金流入（流出）', '折舊費用', '基本每股盈餘合計',
                    '每月營收', '單月營收年增率', '月每股營收', '單月每股營收年增率', 'pretax_profit_margin',
                    'ocf_to_pretax', 'estimated_fcf', 'debt_to_assets', 'interest_coverage',
                    'revenue_qoq', 'asset_turnover', 'pretax_eps', 'cash_per_share',
                    'ocf_proportion', 'pe_ratio', 'pb_ratio', 'cash_to_assets',
                    'net_interest', 'interest_earning_ratio', 'depreciation_to_revenue',
                    'revenue_growth_momentum', 'eps_momentum', 'PctChg_20',
                    'ER_20', 'TrendArea_20', 'rsi_20', 'vr_20', 'bias_20']
    label_column = 'label'
    train_start_time = '2016-01-01'
    val_start_time = '2021-01-01'
    test_start_time = '2024-01-01'
    res_path = '遍历结果.txt'

    # 获取参数
    para_list = GetParas(feature_cols)
    print(len(para_list))

    start = time.perf_counter()
    PoolNum = 10
    pool = Pool(min(len(para_list), PoolNum))
    for idx, para in enumerate(para_list):
        pool.apply_async(main, args=(data_path, label_column,
                                     train_start_time, val_start_time, test_start_time,
                                     res_path, para, idx))
    pool.close()
    pool.join()
    print('总耗时：', time.perf_counter() - start)


