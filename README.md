股票收益预测模型 - GBDT方法实现

📊 项目概述

本项目基于GBDT（梯度提升决策树）算法，结合价量数据和财务数据，构建股票收益预测模型。模型目标为预测股票未来20天收益是否超过所有股票平均收益。

📁 项目结构

1calc_features_fund.py:读取价量数据和财务数据，进行预处理后计算特征指标，包含技术指标和财务指标。

（1）该程序读取data文件夹下taiwan_stock_price_202511122027.csv和reports_202511122033.csv两个表格（需将[https://tripintl.sg.larksuite.com/file/TaHjbTGsRoeRLYxbvvylf5XAgNS](https://tripintl.sg.larksuite.com/file/TaHjbTGsRoeRLYxbvvylf5XAgNS)处压缩包中的两个表格解压到data文件夹下），

（2）对基本面数据进行整理，

（3）计算基本面衍生指标，

（4）将基本面指标和量价数据进行合并，

（5）最后计算技术指标和预测模型的label，

（6）对特征数据进行绝对值标准化和横截面标准化，

（7）保存处理好的数据到processed_data.csv

2训练回测参数寻优多进程.py：调用train.py进行训练，进行参数优化，并将训练结果保存到遍历结果.txt文件中。

3优化结果整理.py:读取遍历结果.txt，提取其中的相关参数与模型预测结果到excel表格中，以供分析并选择参数。

train.py：读取processed_data.csv，进行训练集/验证集/测试集分割，并根据设定好的参数训练模型，最后分别对三个数据集进行预测，并统计预测准确率等预测指标。

model.pkl：保存的训练好的模型文件。


📈 核心特征体系

1. 💰 现金流特征

经营现金流 - 營業活動之淨現金流入（流出）

估算自由现金流 - estimated_fcf

现金流质量 - ocf_to_pretax

2. 📊 盈利能力特征

税前利润率 - pretax_profit_margin

资产周转率 - asset_turnover

3. 🏷️ 估值特征

市盈率 - pe_ratio

市净率 - pb_ratio

4. 📈 增长特征

营收增长率 - 單月營收年增率

增长动量 - revenue_growth_momentum

5. ⚖️ 偿债能力

资产负债率 - debt_to_assets

利息保障倍数 - interest_coverage

6. 📋 每股指标

每股现金 - cash_per_share

每股税前利润 - pretax_eps

7. 📉 技术指标

PctChg_20 - 过去20日涨跌幅

ER_20 - 过去20日涨跌效率系数

TrendArea_20 - 过去20日均线与收盘价围成面积

🔧 训练配置

数据集划分

数据集	时间范围	样本数量

训练集	2016-01-01 至 2020-12-31	116,618

验证集	2021-01-01 至 2023-12-31	79,094

测试集	2024-01-01 至 最新	44,099

参数设置

最终模型参数已在 train.py 中优化配置，直接运行即可获得最佳模型。

🎯 模型性能

准确率表现

数据集	准确率

训练集	65.77%

验证集	53.09%

测试集	53.55%

详细评估报告

训练集评估

              precision    recall  f1-score   support

           0       0.68      0.60      0.64     58309
           1       0.64      0.72      0.68     58309

    accuracy                           0.66    116618
   macro avg       0.66      0.66      0.66    116618
weighted avg       0.66      0.66      0.66    116618

验证集评估

              precision    recall  f1-score   support

           0       0.53      0.48      0.50     39547
           1       0.53      0.59      0.56     39547

    accuracy                           0.53     79094
   macro avg       0.53      0.53      0.53     79094
weighted avg       0.53      0.53      0.53     79094

测试集评估

              precision    recall  f1-score   support

           0       0.63      0.50      0.56     25739
           1       0.45      0.58      0.51     18360

    accuracy                           0.54     44099
   macro avg       0.54      0.54      0.53     44099
weighted avg       0.56      0.54      0.54     44099

🚀 快速开始

1. 数据准备

下载数据压缩包：数据下载链接

解压到 data/ 文件夹下

2. 运行步骤
bash
# 1. 特征计算与数据处理
python calc_features_fund.py

# 2. 参数优化（可选）
python 训练回测参数寻优多进程.py

# 3. 模型训练与评估
python train.py
3. 结果分析
bash
# 整理优化结果
python 优化结果整理.py

📝 注意事项

确保所有依赖库已正确安装

数据文件路径需保持正确

模型训练可能需要较长时间，建议使用多进程版本进行参数优化

📊 结果解读

模型在训练集上表现出较好的学习能力

验证集和测试集准确率均在53%以上，显示出一定的泛化能力

