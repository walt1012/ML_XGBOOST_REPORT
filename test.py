import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('train.csv')

train, test = train_test_split(dataset, test_size=0.2, random_state=7)
feature_name = [x for x in train.columns if x not in ['Survived']]
dtrain = xgb.DMatrix(train[feature_name], label=train['Survived'])
dtest = xgb.DMatrix(test[feature_name], label=test['Survived'])

dtrain.num_col()  # 获得矩阵的列数
dtrain.num_row()  # 获得矩阵的行数
dtrain.get_label()  # 获得矩阵的标签

# 自定义参数
params = {
    'booster': 'gbtree',  # 选择基分类器，gbtree: tree-based models/gblinear: linear models
    'objective': 'binary:logistic',  # 定义最小化损失函数类型
    'eval_metric': ['auc', 'logloss'],  # 验证数据的评估指标，将根据目标分配默认指标（回归的均方根值，分类的误差，排名的平均平均精度）
    'tree_method': 'exact',  # 树构建算法
    'colsample_bytree': 0.7,  # 是构造每棵树时列的子样本比率。对每一个构造的树进行一次二次采样
    'colsample_bylevel': 0.7,  # 是每个级别的列的子采样率。对于树中达到的每个新深度级别，都会进行一次二次采样。从为当前树选择的一组列中对列进行子采​​样
    'gamma': 0.1,  # 后剪枝时，用于控制是否后剪枝的参数
    'min_child_weight': 1,  # 这个参数默认是 1，子节点中需要的最小实例权重（hessian）总和
    'max_depth': 4,  # 每颗树的最大深度，树高越深，越容易过拟合
    'lambda': 2,  # L2正则化权重项。增加此值将使模型更加保守
    'subsample': 0.7,  # 训练实例的子样本比率。将其设置为0.5意味着XGBoost将在树木生长之前随机采样一半的训练数据。这样可以防止过拟合。二次采样将在每个增强迭代中进行一次
    'eta': 0.1,  # 用于更新叶子节点权重时，乘以该系数，避免步长过大。参数值越大，越可能无法收敛。把学习率 eta 设置的小一些，小学习率可以使得后面的学习更加仔细
    'seed': 7,  # 随机数种子
    'nthread': 4  # 线程数
}

watchlist = [(dtest, 'eval'), (dtrain, 'train')]
model = xgb.train(params, dtrain, 100, watchlist)

# 模型保存
model.save_model('xgb_model')
# 模型加载
model = xgb.Booster()
model.load_model('xgb_model')

test['pre'] = model.predict(dtest)
test.to_csv('test_result_by_xgb_model.csv', index=None, header=None)

# 'weight':在所有树分裂过程中，使用该特征分裂的次数
model.get_score(importance_type='weight')
# 'gain':在所有树分裂过程中，使用该特征分裂的平均增益
model.get_score(importance_type='gain')
# 'cover':在所有树分裂过程中，使用该特征分裂的平均覆盖率
model.get_score(importance_type='cover')

xgb.plot_importance(model)
xgb.plot_tree(model)


def ceate_feature_map(features):
    outfile = open('./xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


ceate_feature_map(dataset.columns)  # 特征名列表

xgb.plot_tree(model, fmap='xgb.fmap')
fig, ax = plt.subplots()
fig.set_size_inches(60, 20)
xgb.plot_importance(model)
xgb.plot_tree(model, ax=ax, fmap='xgb.fmap')
fig.savefig('xgb_tree.png')
