# -
使用价量数据和财务数据建立gbdt模型预测股票未来20天收益是否大于所有股票平均收益

脚本文件功能说明：
1calc_features_fund.py:读取价量数据和财务数据，进行预处理后计算特征指标，包含技术指标和财务指标。
（1）该程序读取data文件夹下taiwan_stock_price_202511122027.csv和reports_202511122033.csv两个表格（需将[https://tripintl.sg.larksuite.com/file/TaHjbTGsRoeRLYxbvvylf5XAgNS](https://tripintl.sg.larksuite.com/file/TaHjbTGsRoeRLYxbvvylf5XAgNS)处压缩包中的两个表格解压到data文件夹下），
（2）对基本面数据进行整理，
（3）计算基本面衍生指标，
（4）将基本面指标和量价数据进行合并，
（5）最后计算技术指标和预测模型的label，
（6）对特征数据进行绝对值标准化和横截面标准化，
（7）保存处理好的数据到processed_data.csv

2训练回测参数寻优多进程.py：读取processed_data.csv，并调用train.py进行训练，进行参数优化，并将训练结果保存到遍历结果.txt文件中。

3优化结果整理.py:读取遍历结果.txt，提取其中的相关参数与模型预测结果到excel表格中，以供分析并选择参数。

train.py：
