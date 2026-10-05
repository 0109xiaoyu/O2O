########################计算指标评价模型#########################
#决策树
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings("ignore")
trainfile_class = 'train_class.csv'  # 已预处理测试数据
train_class = pd.read_csv(trainfile_class)
x_train = train_class.drop(['user_id', 'merchant_id', 'coupon_id',
                            'date_received', 'date'], axis=1)

# 将训练样本划分训练样本和验证样本
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_train.iloc[:, :-1],
                                                        x_train.iloc[:, -1],
                                                        test_size=0.3,
                                                        random_state=10)

# 决策树建模
model_dt_evaluate = DecisionTreeClassifier(max_leaf_nodes=16,
                                           random_state=123).fit(x_train1,
                                                                 y_train1)
model_dt_pre = model_dt_evaluate.predict(x_test1)  # 预测结果

# 决策树模型评价指标值
print(metrics.classification_report(y_test1, model_dt_pre))
dt_evaluate_accuracy = accuracy_score(y_test1, model_dt_pre)
print('准确率为%.2f%%:' % (dt_evaluate_accuracy * 100.0))
dt_evaluate_p = precision_score(y_test1, model_dt_pre)
print('精确率为%.2f%% ' % (dt_evaluate_p * 100.0))
dt_evaluate_recall = recall_score(y_test1, model_dt_pre)
print('召回率为%.2f%% ' % (dt_evaluate_recall * 100.0))
dt_evaluate_fl = f1_score(y_test1, model_dt_pre)
print('F1值为%.2f%% ' % (dt_evaluate_fl * 100.0))

#xgboost
import xgboost as xgb

model_xgb_evaluate = xgb.XGBClassifier(max_depth=8, learning_rate=0.1,
                                       n_estimators=160, silent=True,
                                       objective='binary:logistic')
model_xgb_evaluate.fit(x_train1, y_train1)

# 对验证样本进行预测
model_xgb_pre = model_xgb_evaluate.predict(x_test1)

# xgboost 模型评价指标
print(metrics.classification_report(y_test1, model_xgb_pre))
xfb_evaluate_accuracy = accuracy_score(y_test1, model_xgb_pre)
print('准确率为:%.2f%%' % (xfb_evaluate_accuracy * 100.0))
xfb_evaluate_p = precision_score(y_test1, model_xgb_pre)
print('精确率为:%.2f%%' % (xfb_evaluate_p * 100.0))
xfb_evaluate_recall = recall_score(y_test1, model_xgb_pre)
print('召回率为:%.2f%%' % (xfb_evaluate_recall * 100.0))
xfb_evaluate_fl = f1_score(y_test1, model_xgb_pre)
print('F1值为:%.2f%%' % (xfb_evaluate_fl * 100.0))

########################绘制属性重要性评分图#########################
from matplotlib import pyplot as plt
from xgboost import plot_importance

# 显示重要指标
plot_importance(model_xgb_evaluate)
plt.show()
