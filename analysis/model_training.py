########################构建决策树分类模型#########################
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings("ignore")

# 读取数据
train_class = pd.read_csv('train_class.csv')
test = pd.read_csv('test_cleaned.csv')

# 定义要排除的列
exclude_train = ['user_id', 'merchant_id', 'coupon_id', 'date_received', 'date', 'class']
exclude_test = ['user_id', 'merchant_id', 'coupon_id', 'date_received']

# 分离特征和标签
x_train = train_class.drop(columns=exclude_train, errors='ignore')
y_train = train_class['class']

x_test = test.drop(columns=exclude_test, errors='ignore')

# 对齐特征列
for col in x_train.columns:
    if col not in x_test.columns:
        x_test[col] = 0
x_test = x_test[x_train.columns]   # 保持顺序一致

# 决策树建模
model_dt1 = DecisionTreeClassifier(max_leaf_nodes=16, random_state=123)
model_dt1.fit(x_train, y_train)

# 预测
pre_dt = model_dt1.predict(x_test)

# 保存预测结果
dt_class = test[['user_id', 'merchant_id', 'coupon_id']].copy()
dt_class['class'] = pre_dt

# 导出数据
dt_class.to_csv('dt_class.csv', index=False)

########################构建XGBoost分类模型#########################
import xgboost as xgb

# xgboost 模型
model_test = xgb.XGBClassifier(max_depth=8, learning_rate=0.1, n_estimators=160,
                               silent=True, objective='binary:logistic')

# 模型训练
model_test.fit(x_train, y_train)

# 模型预测
y_pred = model_test.predict(x_test)

# DataFrame 存放 xgboost 预测结果
xg_b_class = test[['user_id', 'merchant_id', 'coupon_id']].copy()
xg_b_class['class'] = y_pred

# 导出数据
xgbfile_pre = 'xgb_class.csv'
xg_b_class.to_csv(xgbfile_pre, index=False)