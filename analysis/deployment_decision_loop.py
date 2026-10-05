# -*- coding: utf-8 -*-
"""
05 部署与决策闭环
阈值权衡 + 模拟反馈 + 增量重训
"""

import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from sklearn.metrics import (precision_score, recall_score, f1_score)
from sklearn.model_selection import GroupShuffleSplit


# ----------------------------
# 1. 加载数据与模型
# ----------------------------
print("加载数据与模型...")
train_class = pd.read_csv('train_class.csv')
test_cleaned = pd.read_csv('test_cleaned.csv')

exclude_cols = ['user_id', 'merchant_id', 'coupon_id',
                'date_received', 'date', 'class']
feature_cols = [c for c in train_class.columns if c not in exclude_cols]

model = joblib.load('xgb_model.pkl')


# ----------------------------
# 2. 测试集预测
# ----------------------------
print("\n对测试集进行预测...")
test_X = test_cleaned.reindex(columns=feature_cols, fill_value=0)
y_pred_proba = model.predict_proba(test_X)[:, 1]


# ----------------------------
# 3. 按预算动态选阈值
# ----------------------------
print("生成投放决策（按预算动态分配）...")

top_pct = 0.20
threshold = np.quantile(y_pred_proba, 1 - top_pct)
y_pred = (y_pred_proba >= threshold).astype(int)

print(f"动态阈值：{threshold:.4f}（覆盖前 {top_pct*100:.0f}% 高潜用户）")
print(f"发券用户数：{y_pred.sum()} / {len(y_pred)}（占比 {y_pred.mean()*100:.2f}%）")

decision = pd.DataFrame({
    'user_id': test_cleaned['user_id'],
    'merchant_id': test_cleaned['merchant_id'],
    'coupon_id': test_cleaned['coupon_id'],
    'pred_proba': y_pred_proba,
    'decision': np.where(y_pred == 1, '发券', '不发券')
})
decision.to_csv('decision_output.csv', index=False)


# ----------------------------
# 4. 在独立验证集上评估不同阈值
# ----------------------------
print("\n========== 策略效果评估（按 user_id 分组切分）==========")

X = train_class[feature_cols]
y = train_class['class']
groups = train_class['user_id']

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=10)
_, val_idx = next(gss.split(X, y, groups=groups))
X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx]
y_val_proba = model.predict_proba(X_val)[:, 1]

print(f"\n{'阈值':<8}{'发券比例':<12}{'精确率':<10}{'召回率':<10}{'F1':<10}")
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_hat = (y_val_proba >= th).astype(int)
    p = precision_score(y_val, y_hat, zero_division=0)
    r = recall_score(y_val, y_hat, zero_division=0)
    f = f1_score(y_val, y_hat, zero_division=0)
    print(f"{th:<8}{y_hat.mean()*100:<12.2f}{p:<10.4f}{r:<10.4f}{f:<10.4f}")

print("\n【业务解读】")
print("低阈值(0.3)：覆盖面广但精确率下降，适合拉新期")
print("高阈值(0.7)：精准但召回率低，适合预算紧张期")
print("中阈值(0.5)：平衡点，作为默认推荐")


# ----------------------------
# 5. 模拟反馈数据（仅演示闭环流程）
# ----------------------------
print("\n生成模拟反馈数据（用于演示闭环重训流程）...")
np.random.seed(42)
base_rate = train_class['class'].mean()
true_labels = (
    np.random.random(len(y_pred_proba)) <
    (base_rate * 0.5 + y_pred_proba * 0.5)
).astype(int)

feedback = pd.DataFrame({
    'user_id': test_cleaned['user_id'],
    'merchant_id': test_cleaned['merchant_id'],
    'coupon_id': test_cleaned['coupon_id'],
    'true_label': true_labels
})
feedback.to_csv('feedback.csv', index=False)
print(f"反馈数据已保存，模拟核销率：{true_labels.mean():.4f}")


# ----------------------------
# 6. 闭环重训
# ----------------------------
print("\n开始闭环重训...")
new_train = test_cleaned.copy()
new_train['class'] = true_labels
new_train = new_train[new_train['coupon_id'].notnull()]

combined_train = pd.concat([train_class, new_train], axis=0, ignore_index=True)
X_combined = combined_train[feature_cols]
y_combined = combined_train['class']

model_retrained = xgb.XGBClassifier(
    max_depth=8, learning_rate=0.1, n_estimators=160,
    objective='binary:logistic', eval_metric='logloss', random_state=123
)
model_retrained.fit(X_combined, y_combined)
joblib.dump(model_retrained, 'xgb_model_retrained.pkl')
print("闭环重训完成，新模型已保存为 xgb_model_retrained.pkl")


# ----------------------------
# 7. 新旧模型对比
# ----------------------------
print("\n========== 新旧模型对比（验证集）==========")
for name, m in [('旧模型', model), ('新模型', model_retrained)]:
    proba = m.predict_proba(X_val)[:, 1]
    y_hat = (proba >= 0.5).astype(int)
    print(f"{name}: 准确率={m.score(X_val, y_val):.4f}, "
          f"精确率={precision_score(y_val, y_hat, zero_division=0):.4f}, "
          f"召回率={recall_score(y_val, y_hat, zero_division=0):.4f}, "
          f"F1={f1_score(y_val, y_hat, zero_division=0):.4f}")

print("\n闭环流程结束。")