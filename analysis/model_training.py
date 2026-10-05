# -*- coding: utf-8 -*-
"""
03 构建模型（含 GroupKFold 防泄漏）
使用 GroupKFold 按 user_id 分组，避免同一用户跨集造成重叠泄漏
"""

import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
import xgboost as xgb

warnings.filterwarnings("ignore")


# ----------------------------
# 1. 读取数据
# ----------------------------
train_class = pd.read_csv('train_class.csv')
test = pd.read_csv('test_cleaned.csv')

exclude_cols = ['user_id', 'merchant_id', 'coupon_id',
                'date_received', 'date', 'class']
feature_cols = [c for c in train_class.columns if c not in exclude_cols]

X = train_class[feature_cols]
y = train_class['class']
groups = train_class['user_id']   # ★ 关键：按 user_id 分组

print(f"特征数：{len(feature_cols)}，样本数：{len(X)}，用户数：{groups.nunique()}")


# ----------------------------
# 2. GroupKFold 防泄漏验证
# ----------------------------
def evaluate_model(model, X, y, groups, n_splits=5):
    """
    用 GroupKFold 做交叉验证
    返回: dict(模型名 -> dict(指标均值))
    """
    gkf = GroupKFold(n_splits=n_splits)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_tr, y_tr)
        y_pred = model_clone.predict(X_val)

        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))

        print(f"  Fold {fold}: acc={metrics['accuracy'][-1]:.4f}, "
              f"prec={metrics['precision'][-1]:.4f}, "
              f"rec={metrics['recall'][-1]:.4f}, "
              f"f1={metrics['f1'][-1]:.4f}")

    return {k: np.mean(v) for k, v in metrics.items()}


print("\n========== 决策树 GroupKFold 交叉验证 ==========")
dt_model = DecisionTreeClassifier(max_leaf_nodes=16, random_state=123)
dt_result = evaluate_model(dt_model, X, y, groups)

print("\n========== XGBoost GroupKFold 交叉验证 ==========")
xgb_model = xgb.XGBClassifier(
    max_depth=8, learning_rate=0.1, n_estimators=160,
    objective='binary:logistic', eval_metric='logloss', random_state=123
)
xgb_result = evaluate_model(xgb_model, X, y, groups)


# ----------------------------
# 3. 对比结果
# ----------------------------
print("\n========== 模型对比（GroupKFold 均值）==========")
result_df = pd.DataFrame({
    'DecisionTree': dt_result,
    'XGBoost': xgb_result,
}).T
print(result_df.round(4))
result_df.to_csv('model_cv_results.csv')


# ----------------------------
# 4. 用全量数据训练最终模型
# ----------------------------
print("\n在全量训练集上训练最终模型...")
final_dt = DecisionTreeClassifier(max_leaf_nodes=16, random_state=123)
final_dt.fit(X, y)

final_xgb = xgb.XGBClassifier(
    max_depth=8, learning_rate=0.1, n_estimators=160,
    objective='binary:logistic', eval_metric='logloss', random_state=123
)
final_xgb.fit(X, y)

joblib.dump(final_dt, 'dt_model.pkl')
joblib.dump(final_xgb, 'xgb_model.pkl')
print("模型已保存：dt_model.pkl, xgb_model.pkl")


# ----------------------------
# 5. 预测测试集
# ----------------------------
test_X = test.reindex(columns=feature_cols, fill_value=0)

dt_pred = final_dt.predict(test_X)
xgb_pred = final_xgb.predict(test_X)

dt_output = test[['user_id', 'merchant_id', 'coupon_id']].copy()
dt_output['class'] = dt_pred
dt_output.to_csv('dt_class.csv', index=False)

xgb_output = test[['user_id', 'merchant_id', 'coupon_id']].copy()
xgb_output['class'] = xgb_pred
xgb_output.to_csv('xgb_class.csv', index=False)

print("测试集预测完成")