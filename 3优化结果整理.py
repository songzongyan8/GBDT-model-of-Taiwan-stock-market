import pandas as pd

path = '遍历结果.txt'
save_path = '遍历结果整理1.xlsx'



# 整理回测结果
results = []
with open(path, 'r') as f1:
    text = f1.read().strip('\n')
    lines = text.split('\n\n\n')
    results = []
    for line in lines:
        tmp = line.split('\n')
        features = eval(tmp[0].strip('\n').strip('features: '))
        dt_params = eval(tmp[1].strip('dt_params: '))
        trn_acc = float(tmp[5].strip('trn_acc: '))
        val_acc = float(tmp[6].strip('val_acc: '))
        test_acc = float(tmp[7].strip('test_acc: '))

        dt_params.update({'features': features,
                          'trn_acc': trn_acc,
                          'val_acc': val_acc,
                          'test_acc': test_acc})

        results.append(dt_params.copy())

results0 = pd.DataFrame(results)
results0.to_excel(save_path, index=False)