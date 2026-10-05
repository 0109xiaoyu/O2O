########################数据清洗########################
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

datafile_train = 'train.csv'
datafile_test = 'test.csv'

data_train = pd.read_csv(datafile_train)
data_test = pd.read_csv(datafile_test)

# 清洗训练集
train_clean = data_train.copy()
# 将 'null' 字符串替换为 NaN
train_clean.iloc[:, 5] = train_clean.iloc[:, 5].apply(lambda x: np.nan if x == 'null' else x)
train_clean.iloc[:, 5] = train_clean.iloc[:, 5].apply(lambda x: None if x == 'null' else x)

train_clean['date_received'] = train_clean['date_received'].astype('str').apply(lambda x: x.split('.')[0])
train_clean['date_received'] = pd.to_datetime(train_clean['date_received'], format='%Y%m%d')

train_clean['date'] = train_clean['date'].astype(str).apply(lambda x: x.split('.')[0])
train_clean['date'] = pd.to_datetime(train_clean['date'], format='%Y%m%d', errors='coerce')

train_clean['discount_rate'] = train_clean['discount_rate'].fillna('null')
def discount(x):
    if ':' in x:
        s = x.split(':')
        return round((int(s[0]) - int(s[1])) / int(s[0]), 2)
    elif x == 'null':
        return np.nan
    else:
        return float(x)
train_clean['discount_rate'] = train_clean['discount_rate'].map(discount)

# 清洗测试集
test_clean = data_test.copy()
test_clean.iloc[:, 5] = test_clean.iloc[:, 5].apply(lambda x: np.nan if x == 'null' else x)
test_clean.iloc[:, 5] = test_clean.iloc[:, 5].apply(lambda x: None if x == 'null' else x)
test_clean['date_received'] = test_clean['date_received'].astype('str').apply(lambda x: x.split('.')[0])
test_clean['date_received'] = pd.to_datetime(test_clean['date_received'], format='%Y%m%d')
test_clean['discount_rate'] = test_clean['discount_rate'].fillna('null')
test_clean['discount_rate'] = test_clean['discount_rate'].map(discount)

# 导出清洗后的数据
train_clean.to_csv('clean_train.csv', index=False)
test_clean.to_csv('clean_test.csv', index=False)

########################指标构建########################
from feature_name1 import feature_name

train_quality = train_clean.copy()
test_quality = test_clean.copy()

data_user, data_merchant, data_coupon = feature_name(train_quality=train_quality)
if 'coupon_id' not in data_coupon.columns:
    data_coupon = data_coupon.reset_index().rename(columns={'index': 'coupon_id'})

data_user.to_csv('data_user.csv', index=False)
data_merchant.to_csv('data_merchant.csv', index=False)
data_coupon.to_csv('data_coupon.csv', index=False)

# 构建训练集特征矩阵
train_merge = pd.merge(data_user, train_quality, on='user_id')
train_merge = pd.merge(train_merge, data_merchant, on='merchant_id')
train_merge = pd.merge(train_merge, data_coupon, on='coupon_id', how='left')
train_merge.iloc[:, -2:] = train_merge.iloc[:, -2:].fillna(0)
train_merge.to_csv('train_cleaned.csv', index=False)

# 构建测试集特征矩阵
test_merge = pd.merge(data_user, test_quality, on='user_id', how='left')
test_merge = pd.merge(test_merge, data_merchant, on='merchant_id', how='left')
test_merge = pd.merge(test_merge, data_coupon, on='coupon_id', how='left')
test_merge = test_merge.fillna(0)
test_merge.to_csv('test_cleaned.csv', index=False)

########################构建用户标签和剔除未领券样本########################
train_merge["class"] = 0
diff_days = (train_merge['date'] - train_merge['date_received']).dt.days
train_merge.loc[diff_days <= 15, 'class'] = 1

# 只保留有领券的记录
train_merge = train_merge[train_merge['coupon_id'].notnull()]
train_merge.to_csv('train_class.csv', index=False)