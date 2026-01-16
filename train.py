
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from imblearn.over_sampling import SMOTE                # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler  # 欠抽样处理库RandomUnderSample
from sklearn.metrics import accuracy_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')


def ProData(data_path, feature_cols, label_column, flag, start_time, end_time):
    df = pd.read_csv(data_path)
    df.index = df['date']

    data = df[feature_cols]
    data['label'] = df[label_column]

    # 丢弃无效值
    data = data.dropna(subset=['label'], how='any')

    # 切割数据
    data = data[data.index >= start_time]
    if end_time is not None:
        data = data[data.index < end_time]

    data0 = data[feature_cols + ['label']]
    data0 = data0.rename(columns={'負債總計': 'Total liabilities',
                                  '營業活動之淨現金流入（流出）': 'Net cash flows from (used in) operating activities',
                                  '投資活動之淨現金流入（流出）': 'Net cash flows from (used in) investing activities',
                                  '籌資活動之淨現金流入（流出）': 'Net cash flows from (used in) financing activities',
                                  '營運產生之現金流入（流出）': 'Cash inflow (outflow) generated from operations',
                                  '折舊費用': 'Depreciation expense',
                                  '基本每股盈餘合計': 'Total basic earnings per share',
                                  '每月營收': 'revenue',
                                  '單月營收年增率': 'growth_rate',
                                  '月每股營收': 'per_share',
                                  '單月每股營收年增率': 'per_share_yoy'})

    # 划分特征与标签
    fea, lab = data0.iloc[:, :-1], data0.iloc[:, -1]
    # 归一化
    # 无效值填充
    fea = fea.fillna(method='ffill').fillna(-1)
    fea[fea.isin([np.inf, -np.inf])] = -1
    # 平衡训练集两类样本的数目
    if flag in ['train', 'val']:
        start = time.perf_counter()
        model_smote = SMOTE(sampling_strategy='auto',
                            random_state=0,
                            k_neighbors=100,
                            n_jobs=9)  # 建立SMOTE模型对象
        # model_smote = RandomUnderSampler(sampling_strategy='auto', random_state=34)
        fea, lab = model_smote.fit_sample(fea, lab)  # 输入数据做过抽样处理
        print(time.perf_counter() - start)
    count = lab.groupby(lab).count()               # 对label做分类汇总
    print(flag, count)
    return(data, fea, lab)


def TrainModel(data_path, feature_cols, label_column,
               train_start_time, val_start_time, test_start_time,
               dt_params, categorical_feature, num_boost_round, early_stopping_rounds):

    df_trn, xtrain, ytrain = ProData(data_path, feature_cols, label_column, 'train',
                                     train_start_time, val_start_time)
    df_val, xval, yval = ProData(data_path, feature_cols, label_column, 'val',
                                 val_start_time, test_start_time)
    df_test, xtest, ytest = ProData(data_path, feature_cols, label_column, 'test',
                                    test_start_time, None)

    # 构建lgb中的Dataset格式，和xgboost中的DMatrix是对应的
    lgb_train = lgb.Dataset(xtrain, ytrain, categorical_feature=categorical_feature)
    lgb_eval = lgb.Dataset(xval, yval, reference=lgb_train, categorical_feature=categorical_feature)
    # lgb_test = lgb.Dataset(xtest, ytest, reference=lgb_train, categorical_feature=categorical_feature)
    # 进行训练
    print('开始训练...')
    gbm = lgb.train(dt_params,
                    lgb_train,
                    num_boost_round=num_boost_round,
                    valid_sets=[lgb_train, lgb_eval],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False,
                    )
    imp = gbm.feature_importance()
    idx = np.argsort(imp)[::-1]
    columns = xtest.columns
    for i in idx:
        print(columns[i], ':', imp[i])
    # 保存模型， 保存模型到文件中
    print('保存模型...')
    pickle.dump(gbm, open('model.pkl', 'wb'))
    # 预测
    print('开始预测...')
    pred_train = gbm.predict(xtrain, num_iteration=gbm.best_iteration)
    # pred_train0 = pd.Series(np.argmax(pred_train, axis=1), index=xtrain.index).astype(float)
    pred_train0 = pd.Series(pred_train > 0.5, index=xtrain.index).astype(float)
    pred_val = gbm.predict(xval, num_iteration=gbm.best_iteration)
    # pred_val0 = pd.Series(np.argmax(pred_val, axis=1), index=xval.index).astype(float)
    pred_val0 = pd.Series(pred_val > 0.5, index=xval.index).astype(float)
    pred_test = gbm.predict(xtest, num_iteration=gbm.best_iteration)
    # pred_test0 = pd.Series(np.argmax(pred_test, axis=1), index=xtest.index).astype(float)
    pred_test0 = pd.Series(pred_test > 0.5, index=xtest.index).astype(float)
    # 评估
    print('开始评估...')
    acc_train = accuracy_score(ytrain, pred_train0)
    acc_val = accuracy_score(yval, pred_val0)
    acc_test = accuracy_score(ytest, pred_test0)

    report_train = classification_report(ytrain, pred_train0)
    report_val = classification_report(yval, pred_val0)
    report_test = classification_report(ytest, pred_test0)
    print('训练集准确率为：{}\n验证集准确率为：{}\n测试集准确率为：{}'.format(acc_train, acc_val, acc_test))
    print('训练集分类评估报告：')
    print(report_train)
    print('验证集分类评估报告：')
    print(report_val)
    print('测试集分类评估报告：')
    print(report_test)
    return(df_val, df_test, yval, ytest, pred_val0, pred_test0, (acc_train, acc_val, acc_test, report_train, report_val, report_test))


if __name__ == '__main__':
    data_path = './processed_data.csv'
    feature_cols = [
                   '負債總計', '營業活動之淨現金流入（流出）',
                   '投資活動之淨現金流入（流出）', '籌資活動之淨現金流入（流出）', '營運產生之現金流入（流出）', '折舊費用',
                   '基本每股盈餘合計', '每月營收', '單月營收年增率', '月每股營收', '單月每股營收年增率',
                   'pretax_profit_margin',
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

    categorical_feature = []
    dt_params = {'max_depth': 127,
                 'min_data_in_leaf': 20,
                 'min_child_weight': 1e-8,
                 'feature_fraction': 0.857142857142857,
                 'feature_fraction_seed': 2,
                 'sub_row': 0.8,
                 'bagging_freq': 5,
                 'lambda_l1': 0.5,
                 'lambda_l2': 0.5,
                 'min_gain_to_split': 0.1,
                 'min_data_per_group': 10,
                 'max_cat_threshold': 32,
                 'cat_smooth': 10,
                 'cat_l2': 10,
                 'max_cat_to_onehot': 4,
                 # 'max_cat_group': 10,
                 # 'Task': 'train', # train, predict, convert_model
                 'boosting': 'gbdt',  # gbdt, rf, dart, goss
                 'num_leaves': 63,  # 31
                 'max_bin': 255,
                 'min_data_in_bin': 3,
                 'top_k': 21,
                 'boost_from_average': False,
                 'sample_weight': None,
                 'learning_rate': 0.005,
                 'objective': 'binary',
                 'metric': ['binary_logloss', 'auc'],
                 'num_classes': 1,
                 'train_metric': True,
                 # 'tree_learner': 'serial',  # serial, feature, data, voting
                 'num_threads': 1,  # 'OpenMP_default'
                 'device': 'cpu',
                 # 'weight': 'zf',
                 'verbose': -1,
                 }
    num_boost_round = 5000
    early_stopping_rounds = 50

    TrainModel(data_path, feature_cols, label_column,
               train_start_time, val_start_time, test_start_time,
               dt_params, categorical_feature, num_boost_round, early_stopping_rounds)









