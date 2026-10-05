# -*- coding: utf-8 -*-
"""
04 模型评价
在独立验证集上评估模型，并绘制特征重要性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import joblib
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report)

# ----- 修复中文显示 -----
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


# ----------------------------
# 1. 划分独立验证集（按 user_id 分组）
# ----------------------------
train_class = pd.read_csv('train_class.csv')

exclude_cols = ['user_id', 'merchant_id', 'coupon_id',
                'date_received', 'date', 'class']
feature_cols = [c for c in train_class.columns if c not in exclude_cols]

X = train_class[feature_cols]
y = train_class['class']
groups = train_class['user_id']

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=10)
train_idx, val_idx = next(gss.split(X, y, groups=groups))

X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

print(f"训练集：{len(X_tr)}，验证集：{len(X_val)}")
print(f"训练集正样本率：{y_tr.mean():.4f}，验证集正样本率：{y_val.mean():.4f}")


# ----------------------------
# 2. 加载已训练模型
# ----------------------------
dt_model = joblib.load('dt_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')


# ----------------------------
# 3. 模型评估
# ----------------------------
def eval_on_val(model, X_val, y_val, name):
    y_pred = model.predict(X_val)
    print(f"\n========== {name} ==========")
    print(classification_report(y_val, y_pred, digits=4))
    return {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0),
    }


dt_metrics = eval_on_val(dt_model, X_val, y_val, "决策树")
xgb_metrics = eval_on_val(xgb_model, X_val, y_val, "XGBoost")

metrics_df = pd.DataFrame({'DecisionTree': dt_metrics, 'XGBoost': xgb_metrics}).T
metrics_df.to_csv('model_eval_results.csv')
print("\n========== 汇总 ==========")
print(metrics_df.round(4))


# ----------------------------
# 4. XGBoost 特征重要性
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(xgb_model, ax=ax, importance_type='gain')
plt.title('XGBoost 特征重要性 (gain)')
plt.tight_layout()
plt.savefig('06_特征重要性.png', dpi=150, bbox_inches='tight')
plt.close()

# 输出特征重要性排名
importance = pd.Series(
    xgb_model.get_booster().get_score(importance_type='gain')
).sort_values(ascending=False)
importance.to_csv('feature_importance.csv')
print("\n特征重要性 Top5：")
print(importance.head(5))