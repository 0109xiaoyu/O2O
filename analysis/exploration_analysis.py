# -*- coding: utf-8 -*-
"""
01 探索性分析
对原始数据进行多维度探索性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform

# ----- 修复中文显示 -----
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


# ----------------------------
# 1. 读取数据
# ----------------------------
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# ----------------------------
# 2. 描述性统计
# ----------------------------
explore_train = data_train.describe(percentiles=[], include='all').T
explore_train['null'] = data_train.isnull().sum()
explore_train = explore_train[['null', 'max', 'min']]
explore_train.columns = ['空值数', '最大值', '最小值']
explore_train.to_csv('explore_train.csv')

explore_test = data_test.describe(percentiles=[], include='all').T
explore_test['null'] = data_test.isnull().sum()
explore_test = explore_test[['null', 'max', 'min']]
explore_test.columns = ['空值数', '最大值', '最小值']
explore_test.to_csv('explore_test.csv')

print("训练集描述性统计：")
print(explore_train)

# ----------------------------
# 3. 各月用户消费次数折线图
# ----------------------------
data1 = pd.concat([data_train, data_test], axis=0, join='outer')
data1['date'] = data1['date'].astype('str').apply(lambda x: x.split('.')[0])
data1['date'] = pd.to_datetime(data1['date'], errors='coerce')
data_month = data1['date'].apply(lambda x: x.month if pd.notna(x) else np.nan)
data_count = data_month.value_counts().sort_index()

fig = plt.figure(figsize=(8, 5))
plt.plot(data_count.index, data_count, color='#0504aa', linewidth=3.0, linestyle='-.')
plt.xlabel('月份')
plt.ylabel('消费次数')
plt.title('2016 年各月用户消费次数')
plt.tight_layout()
plt.savefig('01_各月消费次数.png', dpi=150, bbox_inches='tight')
plt.close()

# ----------------------------
# 4. 各月领券次数与领券消费次数柱状图
# ----------------------------
data1['date_received'] = data1['date_received'].astype('str').apply(lambda x: x.split('.')[0])
data1['date_received'] = pd.to_datetime(data1['date_received'], format='%Y%m%d', errors='coerce')
received_month = data1['date_received'].apply(lambda x: x.month if pd.notna(x) else np.nan)
month_count = received_month.value_counts().sort_index()

cop_distance = data1.loc[data1['date'].notnull() & data1['coupon_id'].notnull(),
                         ['user_id', 'distance', 'date', 'discount_rate']]
date_month = cop_distance['date'].apply(lambda x: x.month if pd.notna(x) else np.nan)
datemonth_count = date_month.value_counts().sort_index()
datemonth_countlist = list(datemonth_count)
datemonth_countlist.append(0)

fig = plt.figure(figsize=(8, 5))
x = [i for i in range(1, 8)]
width = 0.4
plt.bar(x, height=list(month_count), width=width, label='用户领券', alpha=0.8, color='#0504aa')
x_shifted = [i + width for i in x]
plt.bar(x_shifted, height=datemonth_countlist, width=width, label='用户领券消费', alpha=0.8, color='skyblue')
plt.legend()
plt.xlabel('月份')
plt.ylabel('次数')
plt.title('2016 年各用户领券次数与领券消费次数')
plt.tight_layout()
plt.savefig('02_领券与消费.png', dpi=150, bbox_inches='tight')
plt.close()

# ----------------------------
# 5. 商户投放优惠券数量 Top10
# ----------------------------
coupon_data = data1.loc[data1['coupon_id'].notnull(), ['merchant_id', 'coupon_id']]
merchant_count = coupon_data['merchant_id'].value_counts()
print('参与发放优惠券商户总数为：', merchant_count.shape[0])
print('最多发放{}张，最少发放{}张'.format(merchant_count.max(), merchant_count.min()))

fig = plt.figure(figsize=(8, 5))
plt.bar(range(len(merchant_count[:10])), height=merchant_count[:10], width=0.5, alpha=0.8, color='#0504aa')
for x, y in enumerate(merchant_count[:10]):
    plt.text(x - 0.4, y + 500, "%s" % y)
plt.xticks(range(len(merchant_count[:10])), merchant_count[:10].index)
plt.xlabel("商户 ID")
plt.ylabel("发放优惠券数量")
plt.title("投放优惠券数量前 10 名的商户 ID")
plt.tight_layout()
plt.savefig('03_商户Top10.png', dpi=150, bbox_inches='tight')
plt.close()

# ----------------------------
# 6. 用户到门店消费距离饼图
# ----------------------------
date_distance = data1.loc[data1['date'].notnull() & data1['distance'].notnull(),
                          ['user_id', 'distance', 'date']]
dis_count = date_distance['distance'].value_counts()

fig = plt.figure(figsize=(7, 7))
plt.pie(x=dis_count, labels=dis_count.index, pctdistance=0.9, autopct='%1.1f%%')
plt.title('用户到门店消费的距离比例')
plt.tight_layout()
plt.savefig('04_消费距离比例.png', dpi=150, bbox_inches='tight')
plt.close()

# ----------------------------
# 7. 持券 vs 未持券消费距离对比
# ----------------------------
cop_distance = data1.loc[data1['date'].notnull() & data1['distance'].notnull() & data1['coupon_id'].notnull(),
                         ['user_id', 'distance', 'date']]
nocop_distance = data1.loc[data1['date'].notnull() & data1['distance'].notnull() & data1['coupon_id'].isnull(),
                           ['user_id', 'distance', 'date']]
cop_count = cop_distance['distance'].value_counts()
nocop_count = nocop_distance['distance'].value_counts()

fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.pie(x=cop_count, labels=cop_count.index, pctdistance=0.9, autopct="%1.1f%%")
plt.title('用户持券到门店消费的距离比例')

plt.subplot(1, 2, 2)
plt.pie(x=nocop_count, labels=nocop_count.index, pctdistance=0.9, autopct="%1.1f%%")
plt.title('用户没用券直接到门店消费的距离比例')
plt.tight_layout()
plt.savefig('05_持券对比.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n探索性分析完成，图表已保存。")