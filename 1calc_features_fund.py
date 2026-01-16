
import datetime
import numpy as np
import pandas as pd
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')


def bs_func(row):
    if row['period'] == 1:
        start_date = datetime.datetime(year=row['year'], month=4, day=1)
        end_date = datetime.datetime(year=row['year'], month=7, day=1)
    elif row['period'] == 2:
        start_date = datetime.datetime(year=row['year'], month=7, day=1)
        end_date = datetime.datetime(year=row['year'], month=10, day=1)
    elif row['period'] == 3:
        start_date = datetime.datetime(year=row['year'], month=10, day=1)
        end_date = datetime.datetime(year=row['year'] + 1, month=1, day=1)
    elif row['period'] == 4:
        start_date = datetime.datetime(year=row['year'] + 1, month=1, day=1)
        end_date = datetime.datetime(year=row['year'] + 1, month=4, day=1)

    res = [start_date, end_date, row['number'], row['type'], row['key'], row['key_en'], row['value'], True]
    return (res)


def cf_func(row):
    if row['period'] == 1:
        start_date = datetime.datetime(year=row['year'], month=4, day=1)
        end_date = datetime.datetime(year=row['year'], month=7, day=1)
    elif row['period'] == 2:
        start_date = datetime.datetime(year=row['year'], month=7, day=1)
        end_date = datetime.datetime(year=row['year'], month=10, day=1)
    elif row['period'] == 3:
        start_date = datetime.datetime(year=row['year'], month=10, day=1)
        end_date = datetime.datetime(year=row['year'] + 1, month=1, day=1)
    elif row['period'] == 4:
        start_date = datetime.datetime(year=row['year'] + 1, month=1, day=1)
        end_date = datetime.datetime(year=row['year'] + 1, month=4, day=1)

    res = [start_date, end_date, row['number'], row['type'], row['key'], row['key_en'], row['value'], False]
    return (res)


def cis_func(row):
    if row['period'] == 1:
        start_date = datetime.datetime(year=row['year'], month=4, day=1)
        end_date = datetime.datetime(year=row['year'], month=7, day=1)
    elif row['period'] == 2:
        start_date = datetime.datetime(year=row['year'], month=7, day=1)
        end_date = datetime.datetime(year=row['year'], month=10, day=1)
    elif row['period'] == 3:
        start_date = datetime.datetime(year=row['year'], month=10, day=1)
        end_date = datetime.datetime(year=row['year'] + 1, month=1, day=1)
    elif row['period'] == 4:
        start_date = datetime.datetime(year=row['year'] + 1, month=1, day=1)
        end_date = datetime.datetime(year=row['year'] + 1, month=4, day=1)

    res = [start_date, end_date, row['number'], row['type'], row['key'], row['key_en'], row['value'], False]
    return (res)


def fs_func(row):
    # 期间数据（flow metrics） - 表示一个时间段内的累计值
    flow_metrics = ['資本支出', '自由現金流', '淨現金流', '母公司業主淨利']
    # 时点数据（stock metrics） - 表示季度末的余额
    stock_metrics = ['長期投資', '短期投資', '應收票據及應收帳款', '附買回票券及債券負債', '預收款項']

    if row['period'] == 1:
        start_date = datetime.datetime(year=row['year'], month=4, day=1)
        end_date = datetime.datetime(year=row['year'], month=7, day=1)
    elif row['period'] == 2:
        start_date = datetime.datetime(year=row['year'], month=7, day=1)
        end_date = datetime.datetime(year=row['year'], month=10, day=1)
    elif row['period'] == 3:
        start_date = datetime.datetime(year=row['year'], month=10, day=1)
        end_date = datetime.datetime(year=row['year'] + 1, month=1, day=1)
    elif row['period'] == 4:
        start_date = datetime.datetime(year=row['year'] + 1, month=1, day=1)
        end_date = datetime.datetime(year=row['year'] + 1, month=4, day=1)

    if row['key'] in flow_metrics:
        res = [start_date, end_date, row['number'], row['type'], row['key'], row['key_en'], row['value'], False]
    elif row['key'] in stock_metrics:
        res = [start_date, end_date, row['number'], row['type'], row['key'], row['key_en'], row['value'], True]
    return (res)


def mr_func(row):
    if row['month'] == 11:
        start_date = datetime.datetime(year=row['year'], month=row['month'] + 1, day=1)
        end_date = datetime.datetime(year=row['year'] + 1, month=1, day=1)
    elif row['month'] == 12:
        start_date = datetime.datetime(year=row['year'] + 1, month=1, day=1)
        end_date = datetime.datetime(year=row['year'] + 1, month=2, day=1)
    else:
        start_date = datetime.datetime(year=row['year'], month=row['month'] + 1, day=1)
        end_date = datetime.datetime(year=row['year'], month=row['month'] + 2, day=1)

    res = [start_date, end_date, row['number'], row['type'], row['key'], row['code'], row['value'], True]
    return (res)


def calc_features(df):
    df_features = df.copy()
    df_features = df_features.sort_values(['stock_id', 'start_date'])

    # 1. 盈利能力特征
    print("\n1. 创建盈利能力特征...")
    if all(col in df.columns for col in ['本期稅前淨利（淨損）', '每月營收']):
        # 税前利润率
        df_features['pretax_profit_margin'] = np.where(df['每月營收'] != 0,
                                                       df['本期稅前淨利（淨損）'] / df['每月營收'],
                                                       np.nan)
        print("  ✓ 税前利润率")
    
    # 2. 现金流特征
    print("\n2. 创建现金流特征...")
    # 经营现金流质量
    if all(col in df.columns for col in ['營業活動之淨現金流入（流出）', '本期稅前淨利（淨損）']):
        df_features['ocf_to_pretax'] = np.where(abs(df['本期稅前淨利（淨損）']) > 0,
                                                df['營業活動之淨現金流入（流出）'] / df['本期稅前淨利（淨損）'],
                                                np.nan)
        print("  ✓ 经营现金流/税前利润")
    
    # 自由现金流估算
    if all(col in df.columns for col in ['營業活動之淨現金流入（流出）', '投資活動之淨現金流入（流出）']):
        df_features['estimated_fcf'] = df['營業活動之淨現金流入（流出）'] + df['投資活動之淨現金流入（流出）']
        print("  ✓ 估算自由现金流")
    
    # 3. 偿债能力特征
    print("\n3. 创建偿债能力特征...")
    if all(col in df.columns for col in ['負債總計', '資產總計']):
        # 资产负债率
        df_features['debt_to_assets'] = np.where(df['資產總計'] != 0,
                                                 df['負債總計'] / df['資產總計'],
                                                 np.nan)
        print("  ✓ 资产负债率")
    
    # 利息保障倍数
    if all(col in df.columns for col in ['本期稅前淨利（淨損）', '利息費用']):
        df_features['interest_coverage'] = np.where(df['利息費用'] != 0,
                                                    df['本期稅前淨利（淨損）'] / abs(df['利息費用']),
                                                    np.nan)
        print("  ✓ 利息保障倍数")
    
    # 4. 增长特征
    print("\n4. 创建增长特征...")
    # 已经有单月营收年增率，可以创建季度环比
    if '每月營收' in df.columns:
        df_features = df_features.sort_values(['stock_id', 'start_date'])
        # 营收季度环比
        df_features['revenue_qoq'] = df_features.groupby('stock_id')['每月營收'].pct_change(3)
        print("  ✓ 营收季度环比")
    
    # 5. 效率特征
    print("\n5. 创建效率特征...")
    if all(col in df.columns for col in ['每月營收', '資產總計']):
        # 资产周转率
        df_features['asset_turnover'] = np.where(df['資產總計'] != 0,
                                                 df['每月營收'] * 12 / df['資產總計'],  # 月营收*12作为年化营收
                                                 np.nan)
        print("  ✓ 资产周转率")
    
    # 6. 每股指标
    print("\n6. 创建每股指标...")
    
    if all(col in df.columns for col in ['本期稅前淨利（淨損）', '當期股數']):
        # 每股税前利润
        df_features['pretax_eps'] = np.where(df['當期股數'] != 0,
                                             df['本期稅前淨利（淨損）'] / df['當期股數'],
                                             np.nan)
        print("  ✓ 每股税前利润")
    
    if all(col in df.columns for col in ['現金及約當現金', '當期股數']):
        # 每股现金
        df_features['cash_per_share'] = np.where(df['當期股數'] != 0,
                                                 df['現金及約當現金'] / df['當期股數'],
                                                 np.nan)
        print("  ✓ 每股现金")
    
    # 7. 现金流结构特征
    print("\n7. 创建现金流结构特征...")
    if all(col in df.columns for col in ['營業活動之淨現金流入（流出）', '投資活動之淨現金流入（流出）', '籌資活動之淨現金流入（流出）']):
        # 经营现金流占比
        total_cf = (df['營業活動之淨現金流入（流出）'].abs() + 
                   df['投資活動之淨現金流入（流出）'].abs() + 
                   df['籌資活動之淨現金流入（流出）'].abs())
        
        df_features['ocf_proportion'] = np.where(total_cf != 0,
                                                 df['營業活動之淨現金流入（流出）'].abs() / total_cf,
                                                 np.nan)
        print("  ✓ 经营现金流占比")
    
    # 8. 估值相关特征
    print("\n8. 创建估值相关特征...")
    if all(col in df.columns for col in ['月均價', '基本每股盈餘合計']):
        # 市盈率
        df_features['pe_ratio'] = np.where(df['基本每股盈餘合計'] != 0,
                                           df['月均價'] / df['基本每股盈餘合計'],
                                           np.nan)
        print("  ✓ 市盈率")
    
    if all(col in df.columns for col in ['月均價', '資產總計', '當期股數']):
        # 市净率（近似）
        book_value_per_share = np.where(df['當期股數'] != 0,
                                        df['資產總計'] / df['當期股數'],
                                        np.nan)
        df_features['pb_ratio'] = np.where(book_value_per_share != 0,
                                           df['月均價'] / book_value_per_share,
                                           np.nan)
        print("  ✓ 市净率")
    
    # 9. 现金相关特征
    print("\n9. 创建现金相关特征...")
    if all(col in df.columns for col in ['現金及約當現金', '資產總計']):
        # 现金资产比
        df_features['cash_to_assets'] = np.where(df['資產總計'] != 0,
                                                 df['現金及約當現金'] / df['資產總計'],
                                                 np.nan)
        print("  ✓ 现金资产比")
    
    # 10. 利息相关特征
    print("\n10. 创建利息相关特征...")
    if all(col in df.columns for col in ['利息收入', '利息費用']):
        # 净利息
        df_features['net_interest'] = df['利息收入'] - df['利息費用']
        print("  ✓ 净利息")
        
        # 利息覆盖能力
        df_features['interest_earning_ratio'] = np.where(df['利息費用'] != 0,
                                                         df['利息收入'] / abs(df['利息費用']),
                                                         np.nan)
        print("  ✓ 利息收入/支出比")
    
    # 11. 折旧特征
    print("\n11. 创建折旧特征...")
    if all(col in df.columns for col in ['折舊費用', '每月營收']):
        # 折旧占营收比
        df_features['depreciation_to_revenue'] = np.where(df['每月營收'] != 0,
                                                          df['折舊費用'] / df['每月營收'],
                                                          np.nan)
        print("  ✓ 折旧/营收比")
    
    # 12. 创建动量特征
    print("\n12. 创建动量特征...")
    if '單月營收年增率' in df.columns:
        # 营收增长动量
        df_features = df_features.sort_values(['stock_id', 'start_date'])
        df_features['revenue_growth_momentum'] = df_features.groupby('stock_id')['單月營收年增率'].rolling(3).mean().reset_index(level=0, drop=True)
        print("  ✓ 营收增长动量（3月平均）")
    
    if '月每股營收' in df.columns:
        # 每股营收动量
        df_features['eps_momentum'] = df_features.groupby('stock_id')['月每股營收'].pct_change(3)
        print("  ✓ 每股营收季度动量")
    
    print(f"\n特征创建完成! 总共创建了 {len(df_features.columns) - len(df.columns)} 个新特征")
    
    return(df_features)


def PctChg(df, period):
    pct_chg = df['close'] / df['close'].shift(period) - 1
    pct_chg = pct_chg.fillna(0)
    return(pct_chg)


def ER(df, period):
    df['net_change'] = df['close'] - df['close'].shift(period, fill_value=0)
    df['total_change'] = abs(df['close'] - df['close'].shift(1, fill_value=0)).rolling(window=period).sum()
    er = np.where(df['total_change'] > 0, df['net_change'] / df['total_change'], 0)
    return(er)


def TrendArea(df, period):
    """
    不论上涨和下跌，把起始点和终点连线，看K线和连线所组成的面积大小，上方面积为负，下方为正。
    上涨加速：chg > 0 and down_area + up_area > 0
    上涨减缓：chg > 0 and down_area + up_area < 0
    下跌加速：chg < 0 and down_area + up_area < 0
    下跌减缓：chg < 0 and down_area + up_area > 0
    chg和area都趋近于0为震荡
    """
    # 计算涨跌幅
    df['chg_pct'] = df['close'] / df['close'].shift(period) - 1
    chg_pct = df['chg_pct'].fillna(df['close'] / df['open'].iloc[0] - 1).values.copy()
    close = df['close'].values.copy()
    close_arr = np.array([0] * period)
    x_nums = np.array(range(period))

    res = []
    for idx, (chg, c) in enumerate(zip(chg_pct, close), start=1):
        # 更新close矩阵
        close_arr[:-1] = close_arr[1:]
        close_arr[-1] = c
        # 归一化
        if idx > period:
            normalized_prices = close_arr / close_arr[0]
        elif 2 < idx <= period:
            tmp_close_arr = close_arr[-idx:]
            normalized_prices = tmp_close_arr / tmp_close_arr[0]
            x_nums = np.array(range(idx))
        else:
            res.append(0)
            continue

        # 计算连线
        start_price = normalized_prices[0]
        end_price = normalized_prices[-1]
        slope = (end_price - start_price) / (len(normalized_prices) - 1)
        intercept = start_price
        line_prices = slope * x_nums + intercept
        # 计算面积
        diff = normalized_prices - line_prices
        up_area = -np.sum(diff[diff > 0])
        down_area = -np.sum(diff[diff < 0])
        area = down_area + up_area
        res.append(area)
    return(res)


def RSI(df, period):
    diff = df['close'].diff(1)
    close_up = diff.where(diff > 0, 0.0)  # 筛选出差值大于0的, 否则就赋值为0
    close_down = diff.where(diff < 0, 0.0).abs()  # 筛选出差值小于0的, 否则就赋值为0
    # close_up_ma=SMA(close_up,N,1),其中SMA(X,N,M)为 SMA=M/N*X+(N-M)/N*REF(SMA,1), 可利用ewm控制传参完成计算SMA
    close_up_ma = close_up.ewm(alpha=1 / period, min_periods=1, adjust=False).mean()
    close_down_ma = close_down.ewm(alpha=1 / period, min_periods=1, adjust=False).mean()
    rsi = 100 * close_up_ma / (close_up_ma + close_down_ma)
    return(rsi)


def VR(df, period):
    df['涨跌幅'] = df['close'].pct_change(1)
    df['AV'] = df['vol'] * df['涨跌幅'].abs() * (df['涨跌幅'] > 0).astype(int)
    df['BV'] = df['vol'] * df['涨跌幅'].abs() * (df['涨跌幅'] < 0).astype(int)
    df['CV'] = df['vol'] * df['涨跌幅'].abs() * (df['涨跌幅'] == 0).astype(int)
    # N天以来，对AV,BV,CV进行求和
    df['AVS'] = df['AV'].rolling(period, min_periods=1).sum()
    df['BVS'] = df['BV'].rolling(period, min_periods=1).sum()
    df['CVS'] = df['CV'].rolling(period, min_periods=1).sum()
    # VR = (AVS + 0.5*CVS)/(BVS + 0.5*CVS)
    VR = (df['AVS'] + 0.5 * df['CVS']) / (df['BVS'] + 0.5 * df['CVS'])

    return(VR)


def Bias(df, length):
    ma = df['close'].rolling(length, min_periods=1).mean()
    ma_dist = df['close'] / ma - 1
    return(ma_dist)


def calc_tech_feas(df):
    for period in [20]:
        df['PctChg_{}'.format(period)] = PctChg(df.copy(), period)

    for period in [20]:
        df['ER_{}'.format(period)] = ER(df.copy(), period)

    for period in [20]:
        df['TrendArea_{}'.format(period)] = TrendArea(df.copy(), period)

    for period in [20]:
        df['rsi_{}'.format(period)] = RSI(df.copy(), period)

    for period in [20]: 
        df['vr_{}'.format(period)] = VR(df.copy(), period)

    for period in [20]: # 10, 20, 30, 40, 50, 60, 80, 100, 120
        df['bias_{}'.format(period)] = Bias(df.copy(), period)

    return(df)


def calc_labels(df, target_period):
    df['rtn'] = df.groupby('stock_id')['close'].pct_change(target_period).shift(-target_period)
    avg_rtn = df.groupby('date')['rtn'].mean().to_dict()
    # df['avg_rtn'] = df['date'].replace(avg_rtn)
    df['avg_rtn'] = df['date'].map(avg_rtn).fillna(0)
    labels = (df['rtn'] > df['avg_rtn']).astype(int)
    return(labels)


def cross_sectional_standardization(df, date_column='date'):
    """
    横截面标准化
    """
    if date_column not in df.columns:
        print("日期列不存在，跳过横截面标准化")
        return df
    
    df_cs = df.copy()
    dates = df_cs[date_column].unique()
    
    print(f"\n开始横截面标准化，共有 {len(dates)} 个交易日")
    
    # 获取数值型特征
    # numeric_cols = df_cs.select_dtypes(include=[np.number]).columns.tolist()
    # identifier_cols = ['stock_id', 'date', 'start_date', 'end_date']
    # numeric_cols = [col for col in numeric_cols if col not in identifier_cols]
    numeric_cols = list(df_cs.columns)[10:]
    
    # 按日期进行横截面标准化
    for date in dates:
        mask = df_cs[date_column] == date
        
        for col in numeric_cols:
            if col == 'label':
                continue 
            values = df_cs.loc[mask, col]
            valid_values = values.dropna()
            
            if len(valid_values) >= 2:
                mean_val = valid_values.mean()
                std_val = valid_values.std()
                
                if std_val > 0:
                    standardized = (values - mean_val) / std_val
                    df_cs.loc[mask, col] = standardized
                else:
                    df_cs.loc[mask, col] = 0
    
    print(f"横截面标准化完成，创建了 {len(numeric_cols)} 个标准化特征")
    return df_cs


if __name__ == '__main__':
    ########################## 参数设定
    # 预测未来多少天的收益
    target_period = 20 
    # 价量数据路径
    price_path = './data/taiwan_stock_price_202511122027.csv'
    # 财务数据路径
    reports_path = './data/reports_202511122033.csv'
    # 处理结果保存路径
    save_path = './processed_data.csv'
    

    ########################## 处理基本面数据
    df = pd.read_csv(reports_path)
    types = df['type'].unique().tolist()
    results = []
    for tp in types:
        tmp_df = df[df['type'] == tp].copy()

        if tp == 'balance_sheet':
            res = tmp_df.apply(lambda x: bs_func(x), axis=1)
        elif tp == 'cash_flow':
            res = tmp_df.apply(lambda x: cf_func(x), axis=1)
        elif tp == 'comprehensive_income_statement':
            res = tmp_df.apply(lambda x: cis_func(x), axis=1)
        elif tp == 'financial_statement':
            res = tmp_df.apply(lambda x: fs_func(x), axis=1)
        elif tp == 'monthly_revenue':
            res = tmp_df.apply(lambda x: mr_func(x), axis=1)

        res0 = pd.DataFrame(res.values.tolist(), columns=['start_date', 'end_date', 'stock_id', 'type', 'key', 'key_en', 'value', 'pro_flag'])
        results.append(res0)

    results0 = pd.concat(results, axis=0)
    # results0.to_csv(save_path, index=False)

    results1 = results0.pivot_table(index=['stock_id', 'start_date'],
                                    columns='key',
                                    values='value',
                                    aggfunc='first').reset_index()
    # results1.to_csv(save_path, index=False)


    ########################## 计算财务衍生特征
    results2 = calc_features(results1)
    features = results2[['stock_id', 'start_date', 
                         '現金及約當現金', '資產總計', '負債總計',
                         '營業活動之淨現金流入（流出）', '投資活動之淨現金流入（流出）',
                         '籌資活動之淨現金流入（流出）', '營運產生之現金流入（流出）',
                         '折舊費用', '基本每股盈餘合計', '每月營收',
                         '單月營收年增率', '月每股營收', '單月每股營收年增率',
                         'pretax_profit_margin', 'ocf_to_pretax', 'estimated_fcf',
                         'debt_to_assets', 'interest_coverage', 'revenue_qoq', 'asset_turnover',
                         'pretax_eps', 'cash_per_share', 'ocf_proportion', 'pe_ratio',
                         'pb_ratio', 'cash_to_assets', 'net_interest', 'interest_earning_ratio',
                         'depreciation_to_revenue', 'revenue_growth_momentum', 'eps_momentum']]
    # features.to_csv(save_path, index=False)
    
    
    ########################## 读取股价数据并插入基本面特征数据
    df = pd.read_csv(price_path)
    df = df.sort_values(['stock_id', 'date'])
    df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x.split()[0], '%Y-%m-%d'))
    
    for col in features.columns[3:]:
        tmp = features[['stock_id', 'start_date', col]].copy()
        tmp = tmp.dropna(subset=[col])
        tmp = tmp.sort_values(by=['stock_id', 'start_date'])
    
        df = pd.merge_asof(df.sort_values('date'),  # 量价数据按日期排序
                           tmp.sort_values('start_date'),  # 财务数据按start_date排序
                           left_on='date',  # 量价数据的日期
                           right_on='start_date',  # 财务数据的报告结束日期
                           by='stock_id',  # 按股票代码合并
                           direction='backward'  # 向后查找，即使用最新的可用财务数据
                                  )
        del df['start_date']
        
        
    ########################## 计算模型标签及技术指标
    df = df.sort_values(['stock_id', 'date'])
    df = df.rename(columns={'trading_volume': 'vol',
                            'trading_money': 'amount',
                            'max': 'high',
                            'min': 'low'})
    # 计算模型标签
    df['label'] = calc_labels(df.copy(), target_period)

    # 计算技术指标
    stock_ids = df['stock_id'].unique().tolist()
    pool = Pool(min(10, len(stock_ids)))
    results = []
    for stock_id in stock_ids:
        tmp_df = df[df['stock_id'] == stock_id].copy()
        tmp_df = tmp_df.drop_duplicates(subset=['date'], keep='first')
        # res = calc_tech_feas(tmp_df)
        res = pool.apply_async(calc_tech_feas, args=(tmp_df,))
        results.append(res)
    pool.close()
    pool.join()

    results = [res.get() for res in results]
    df = pd.concat(results, axis=0)


    ########################## 特征数据标准化处理
    # 绝对值标准化
    absolute_features = ['現金及約當現金', '負債總計',
                         '營業活動之淨現金流入（流出）', '投資活動之淨現金流入（流出）',
                         '籌資活動之淨現金流入（流出）',
                         '營運產生之現金流入（流出）', '折舊費用', '每月營收']
    
    for feature in absolute_features:
        if feature in df.columns:
            df[feature] = df[feature] / df['資產總計']
    
    del df['資產總計']
    
    
    # 横截面标准化
    df = df.dropna(how='any', axis=0)
    df = cross_sectional_standardization(df, date_column='date')
    df.to_csv(save_path, index=False)
    










































