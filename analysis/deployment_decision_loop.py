import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

# ----------------------------
# 1. 加载数据与模型
# ----------------------------
print("加载数据与模型...")
train_class = pd.read_csv('train_class.csv')
test_cleaned = pd.read_csv('test_cleaned.csv')

exclude_train = ['user_id', 'merchant_id', 'coupon_id', 'date_received', 'date', 'class']
feature_cols = [c for c in train_class.columns if c not in exclude_train]

model_path = 'xgb_model.pkl'
if os.path.exists(model_path):
    print("加载已有模型...")
    model = joblib.load(model_path)
else:
    print("未找到模型，开始训练...")
    model = xgb.XGBClassifier(
        max_depth=8, learning_rate=0.1, n_estimators=160,
        objective='binary:logistic', eval_metric='logloss'
    )
    model.fit(train_class[feature_cols], train_class['class'])
    joblib.dump(model, model_path)

# ----------------------------
# 2. 对新数据进行预测
# ----------------------------
print("\n对测试集进行预测...")
X_test = test_cleaned[feature_cols]
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ----------------------------
# 3. 策略优化：不是简单 0/1，而是按预算/比例动态选阈值
# ----------------------------
print("生成投放决策（按预算分配，而非固定阈值）...")

# 假设每月预算允许发放 top 20% 的高潜用户，而不是固定阈值 0.5
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
# 4. 用训练集上的验证结果展示策略效果
#    （不再用"假反馈"，而是用真实的训练集反推策略价值）
# ----------------------------
print("\n========== 策略效果评估（基于训练集）==========")
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    train_class[feature_cols], train_class['class'],
    test_size=0.3, random_state=10, stratify=train_class['class']
)
y_val_proba = model.predict_proba(X_val)[:, 1]

# 对比不同阈值策略
print(f"\n{'阈值':<8}{'发券比例':<12}{'精确率':<10}{'召回率':<10}{'F1':<10}")
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_hat = (y_val_proba >= th).astype(int)
    p = precision_score(y_val, y_hat, zero_division=0)
    r = recall_score(y_val, y_hat, zero_division=0)
    f = f1_score(y_val, y_hat, zero_division=0)
    print(f"{th:<8}{y_hat.mean()*100:<12.2f}{p:<10.4f}{r:<10.4f}{f:<10.4f}")

# 业务解读
print("\n【业务解读】")
print("低阈值(0.3)：覆盖面广但精确率下降，适合拉新期")
print("高阈值(0.7)：精准但召回率低，适合预算紧张期")
print("中阈值(0.5)：平衡点，作为默认推荐")

# ----------------------------
# 5. 反馈数据生成（仅演示闭环流程，非真实评估）
# ----------------------------
print("\n生成模拟反馈数据（用于演示闭环重训流程）...")
np.random.seed(42)
# 用基准核销率 + 特征驱动生成，而非直接copy预测概率
base_rate = train_class['class'].mean()
true_labels = (np.random.random(len(y_pred_proba)) <
               (base_rate * 0.5 + y_pred_proba * 0.5)).astype(int)

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
    objective='binary:logistic', eval_metric='logloss'
)
model_retrained.fit(X_combined, y_combined)
joblib.dump(model_retrained, 'xgb_model_retrained.pkl')
print("闭环重训完成，新模型已保存为 xgb_model_retrained.pkl")

# ----------------------------
# 7. 新旧模型对比（在验证集上）
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