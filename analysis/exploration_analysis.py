########################获取数据########################
import pandas as pd

# 文件路径
datafile_train = 'train.csv'    # train 原始数据,第一行为属性标签
datafile_test = 'test.csv'    # test 原始数据,第一行为属性标签

# 读取训练样本和测试样本
data_train = pd.read_csv(datafile_train)
data_test = pd.read_csv(datafile_test)

########################描述性统计########################
import numpy as np
import pandas as pd

result_train = 'explore_train.csv'  # 训练样本的描述性统计分析
result_test = 'explore_test.csv'  # 测试样本的描述性统计分析
# 训练样本的描述性统计分析
explore_train = data_train.describe(percentiles = [], include = 'all').T
# percentiles 参数是指定计算多少的分位数表（如 1/4 分位数、中位数等）

# 计算缺失值
explore_train['null'] = data_train.isnull().sum()
explore_train = explore_train[['null', 'max', 'min']]
explore_train.columns = ['空值数', '最大值', '最小值']  # 表头重命名

# 测试样本的描述性统计分析
explore_test = data_test.describe(percentiles = [], include = 'all').T
explore_test['null'] = data_test.isnull().sum()  # 统计缺失值
explore_test = explore_test[['null', 'max', 'min']]
explore_test.columns = ['空值数', '最大值', '最小值']  # 表头重命名

# 导出结果
explore_train.to_csv(result_train)
explore_test.to_csv(result_test)
explore_test

########################绘制折线图分析用户消费次数########################
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import platform

data1 = pd.concat([data_train, data_test],axis = 0,join ='outer')
# 处理 date 字段
data1['date'] = data1['date'].astype('str').apply(lambda x:x.split('.')[0])
data1['date'] = pd.to_datetime(data1['date']) # 将 date 转为 datetime 类型
# 提取月份
data_month = data1['date'].apply(lambda x : x.month)
# 对各月份用户消费次数进行统计
data_count = data_month.value_counts().sort_index(ascending=True)
data_count
# 绘制用户消费次数折线图
fig = plt.figure(figsize=(8, 5)) # 设置画布大小

# ----- 修复中文显示 -----
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']          # Windows 黑体
elif system == 'Darwin':                                  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC']     # 苹方
else:                                                     # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

plt.plot(data_count.index, data_count, color='#0504aa',
    linewidth=3.0, linestyle='-.')

plt.xlabel('月份')
plt.ylabel('消费次数')
plt.title('2016 年各月用户消费次数')
plt.show()
plt.close()

########################绘制柱形图分析用户领券数与领券消费数########################
# 处理 data_received 字段
data1['date_received'] = data1['date_received'].astype(
    'str').apply(lambda x: x.split('_')[0].split('.')[0])   # 先取下划线前，再取小数点前
data1['date_received'] = pd.to_datetime(data1['date_received'], format='%Y%m%d')  # 将 date 转为 datetime 类型，注意格式
# 提取领券日期的月份
received_month = data1['date_received'].apply(lambda x: x.month)
month_count = received_month.value_counts().sort_index(ascending=True)
# 获取领券消费数据
cop_distance = data1.loc[data1['date'].notnull() & data1[
    'coupon_id'].notnull(), ['user_id', 'distance', 'date', 'discount_rate']]
# 统计 1~7 月领券消费次数
date_month = cop_distance['date'].apply(lambda x: x.month)
datemonth_count = date_month.value_counts().sort_index(ascending=True)
datemonth_countlist = list(datemonth_count)  # 转为列表
datemonth_countlist.append(0)  # 列表末尾追加一个数字 0
# 给用户领取次数与领券消费次数的柱形图
import matplotlib.pyplot as plt
import platform

fig = plt.figure(figsize=(8, 5))  # 设置画布大小

# ----- 修复中文显示-----
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']          # Windows 黑体
elif system == 'Darwin':                                  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC']     # 苹方
else:                                                     # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

name_list = [i for i in range(1,7)]; x = [i for i in range(1,8)]
width = 0.4; alpha = 0.8  # width 设置宽度大小,alpha 为透明度
plt.bar(x,height = list(month_count),
    width = width,label='用户领券',alpha=alpha,color='#0504aa')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x,height = datemonth_countlist,
    width = width,label='用户领券消费',alpha=alpha,color='skyblue')
plt.legend()  # 图例
plt.xlabel('月份')
plt.ylabel('次数')
plt.title('2016 年各用户领券次数与领券消费次数')
plt.show()

########################绘制柱形图分析商户投放优惠券数量########################
# 提取商户发放优惠券数据
coupon_data = data1.loc[data1['coupon_id'].notnull(), ['merchant_id', 'coupon_id']]
merchant_count = coupon_data['merchant_id'].value_counts()
print('参与发放优惠券商户总数为：', merchant_count.shape[0])
print('商户最多发放优惠券数{}张，最少发放优惠券数{}张'.format(merchant_count.max(), merchant_count.min()))

# 绘制柱形图分析商家投放数量
import matplotlib.pyplot as plt
import platform

fig = plt.figure(figsize=(8, 5))  # 设置画布大小

# ----- 修复中文显示-----
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']          # Windows 黑体
elif system == 'Darwin':                                  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC']     # 苹方
else:                                                     # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

plt.bar(range(len(merchant_count[:10])),
        height=merchant_count[:10], width=0.5,
        alpha=0.8, color='#0504aa')

# 给条形图添加数据标注
for x, y in enumerate(merchant_count[:10]):
    plt.text(x - 0.4, y + 500, "%s" % y)
plt.xticks(range(len(merchant_count[:10])), merchant_count[:10].index)
plt.xlabel("商户 ID")
plt.ylabel("发放优惠券数量")
plt.title("投放优惠券数量前 10 名的商户 ID")
plt.show()
plt.close()

########################绘制饼图分析用户到门店消费距离########################
# 提取用户消费次数数据
date_distance = data1.loc[data1['date'].notnull() & data1[
    'distance'].notnull(), ['user_id', 'distance', 'date']]
print('数据形状:', date_distance.shape)

# 统计用户消费次数
dis_count = date_distance['distance'].value_counts()
# 绘制用户到门店消费的距离比例饼图
import matplotlib.pyplot as plt
import platform

fig = plt.figure(figsize=(7, 7))  # 设置画布大小

# ----- 修复中文显示（自动适配操作系统）-----
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']          # Windows 黑体
elif system == 'Darwin':                                  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC']     # 苹方
else:                                                     # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=14)

plt.pie(x=dis_count, labels=dis_count.index,
        pctdistance=0.9, autopct='%1.1f%%')
# pctdistance 是数据标签的距离圆心位置,取值范围为 0~1

plt.title('用户到门店消费的距离比例')
plt.show()
plt.close()

########################绘制饼图分析距离########################
# 提取用户领券到店铺消费距离数据
cop_distance = data1.loc[data1['date'].notnull() & data1['distance'].notnull() & data1['coupon_id'].notnull(),
                         ['user_id', 'distance', 'date', 'discount_rate']]
print('数据形状:', cop_distance.shape)
cop_count = cop_distance['distance'].value_counts()

# 提取用户未用券到店铺消费距离数据
nocop_distance = data1.loc[data1['date'].notnull() & data1['distance'].notnull() & data1['coupon_id'].isnull(),
                           ['user_id', 'distance', 'date', 'discount_rate']]
print('数据形状:', nocop_distance.shape)
nocop_count = nocop_distance['distance'].value_counts()

# 绘制用户持券到门店消费的距离比例饼图
import matplotlib.pyplot as plt
import platform

# ----- 修复中文显示-----
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']          # Windows 黑体
elif system == 'Darwin':                                  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC']     # 苹方
else:                                                     # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

fig = plt.figure(figsize=(12, 6))  # 设置画布大小

# 子图1：持券消费距离比例
plt.subplot(1, 2, 1)
plt.pie(x=cop_count, labels=cop_count.index, pctdistance=0.9, autopct="%1.1f%%")
plt.title('用户持券到门店消费的距离比例')

# 子图2：未持券消费距离比例
plt.subplot(1, 2, 2)
plt.pie(x=nocop_count, labels=nocop_count.index, pctdistance=0.9, autopct="%1.1f%%")
plt.title('用户没用券直接到门店消费的距离比例')

plt.show()
plt.close()