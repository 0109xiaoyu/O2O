# -*- coding: utf-8 -*-
"""
02 数据预处理与特征工程
清洗原始数据、构建特征矩阵、生成用户标签
"""

import warnings
import pandas as pd
import numpy as np

from feature_engineering import (
    build_user_features, build_merchant_features,
    build_coupon_features, merge_features
)

warnings.filterwarnings("ignore")


# ----------------------------
# 1. 读取数据
# ----------------------------
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')


# ----------------------------
# 2. 数据清洗
# ----------------------------
def clean(df, is_train=True):
    df = df.copy()

    # 将 'null' 字符串替换为 NaN
    df.iloc[:, 5] = df.iloc[:, 5].apply(lambda x: np.nan if x == 'null' else x)

    # 解析 date_received
    df['date_received'] = df['date_received'].astype('str').apply(lambda x: x.split('.')[0])
    df['date_received'] = pd.to_datetime(df['date_received'], format='%Y%m%d', errors='coerce')

    if is_train:
        # 解析 date（仅训练集有）
        df['date'] = df['date'].astype(str).apply(lambda x: x.split('.')[0])
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')

    # 处理 discount_rate
    df['discount_rate'] = df['discount_rate'].fillna('null')

    def parse_discount(x):
        if isinstance(x, str) and ':' in x:
            s = x.split(':')
            return round((int(s[0]) - int(s[1])) / int(s[0]), 2)
        elif x == 'null':
            return np.nan
        else:
            try:
                return float(x)
            except (ValueError, TypeError):
                return np.nan

    df['discount_rate'] = df['discount_rate'].map(parse_discount)
    return df


train_clean = clean(data_train, is_train=True)
test_clean = clean(data_test, is_train=False)

train_clean.to_csv('clean_train.csv', index=False)
test_clean.to_csv('clean_test.csv', index=False)

print("数据清洗完成")


# ----------------------------
# 3. 构建特征矩阵
# ----------------------------
data_user = build_user_features(train_clean)
data_merchant = build_merchant_features(train_clean)
data_coupon = build_coupon_features(train_clean)

data_user.to_csv('data_user.csv')
data_merchant.to_csv('data_merchant.csv')
data_coupon.to_csv('data_coupon.csv')

# 训练集特征矩阵
train_merge = merge_features(train_clean, data_user, data_merchant, data_coupon)
train_merge.to_csv('train_cleaned.csv', index=False)

# 测试集特征矩阵
test_merge = merge_features(test_clean, data_user, data_merchant, data_coupon)
test_merge.to_csv('test_cleaned.csv', index=False)

print("特征矩阵构建完成，训练集维度：", train_merge.shape, "，测试集维度：", test_merge.shape)


# ----------------------------
# 4. 构建标签 & 剔除未领券样本
# ----------------------------
# ★ 强制日期类型，避免 merge 后类型被污染
train_merge['date_received'] = pd.to_datetime(train_merge['date_received'], errors='coerce')
train_merge['date'] = pd.to_datetime(train_merge['date'], errors='coerce')

train_merge['class'] = 0
diff_days = (train_merge['date'] - train_merge['date_received']).dt.days
train_merge.loc[diff_days <= 15, 'class'] = 1

# 只保留有领券的记录
train_merge = train_merge[train_merge['coupon_id'].notnull()]
train_merge.to_csv('train_class.csv', index=False)

print("标签构建完成，正样本率：{:.4f}".format(train_merge['class'].mean()))
print("训练集最终样本数：", len(train_merge))