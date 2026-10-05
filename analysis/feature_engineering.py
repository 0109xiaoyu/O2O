# -*- coding: utf-8 -*-
"""
特征工程模块
从用户、商户、优惠券三个维度构建 13 个特征
"""

import pandas as pd
import numpy as np


def build_user_features(df):
    """构建用户维度特征（5个）"""
    data_user = pd.DataFrame()

    # 用户使用优惠券消费次数
    data_user['user_use_coupon_times'] = (
        (df[['date_received', 'date']].count(axis=1) == 2)
        .groupby(df['user_id']).sum()
    )

    # 用户总消费次数
    data_user['user_consume_times'] = (
        df.loc[:, 'date':'date'].count(axis=1)
        .groupby(df['user_id']).sum()
    )

    # 用户使用优惠券消费次数与总消费次数的比值
    data_user['user_use_coupon_rate'] = (
        data_user['user_use_coupon_times'] / data_user['user_consume_times']
    )
    data_user['user_use_coupon_rate'] = data_user['user_use_coupon_rate'].fillna(0)

    # 用户领取优惠券而未使用的数量
    data_user['user_receive_coupon_unused_times'] = (
        (df['coupon_id'].notnull() & df['date'].isnull())
        .groupby(df['user_id']).sum()
    )

    # 用户使用优惠券的日期平均与领取日期相隔多少天
    data_user['user_mean_use_coupon_interval'] = (
        (df['date'] - df['date_received']).dt.days
        .groupby(df['user_id']).mean()
    )
    data_user['user_mean_use_coupon_interval'] = (
        data_user['user_mean_use_coupon_interval']
        .fillna(data_user['user_mean_use_coupon_interval'].max() + 1)
    )

    return data_user


def build_merchant_features(df):
    """构建商户维度特征（5个）"""
    data_merchant = pd.DataFrame()

    # 商户发放的优惠券被使用的数量
    data_merchant['merchant_launch_coupon_used_count'] = (
        (df[['date_received', 'date']].count(axis=1) == 2)
        .groupby(df['merchant_id']).sum()
    )

    # 商户发放的优惠券被使用数与商户总消费次数的比值
    merchant_consume_times = (
        df.loc[:, 'date':'date'].count(axis=1)
        .groupby(df['merchant_id']).sum()
    )
    data_merchant['merchant_launch_coupon_used_rate'] = (
        data_merchant['merchant_launch_coupon_used_count'] / merchant_consume_times
    )
    data_merchant['merchant_launch_coupon_used_rate'] = (
        data_merchant['merchant_launch_coupon_used_rate'].fillna(0)
    )

    # 商户发放优惠券的数量
    data_merchant['merchant_launch_coupon_count'] = (
        df.loc[:, 'coupon_id':'coupon_id'].count(axis=1)
        .groupby(df['merchant_id']).sum()
    )

    # 商户发放优惠券而未被使用的数量
    data_merchant['merchant_receive_unused'] = (
        (df['coupon_id'].notnull() & df['date'].isnull())
        .groupby(df['merchant_id']).sum()
    )

    # 商户发放的优惠券平均相隔多少天会被使用
    data_merchant['merchant_mean_launch_coupon_interval'] = (
        (df['date'] - df['date_received']).dt.days
        .groupby(df['merchant_id']).mean()
    )
    data_merchant['merchant_mean_launch_coupon_interval'] = (
        data_merchant['merchant_mean_launch_coupon_interval']
        .fillna(data_merchant['merchant_mean_launch_coupon_interval'].max() + 1)
    )

    return data_merchant


def build_coupon_features(df):
    """构建优惠券维度特征（3个）"""
    data_coupon = pd.DataFrame()

    # 15天内核销的优惠券数量（优惠券流行度）
    day_fifteen = (df['date'] - df['date_received']).dt.days <= 15
    data_coupon['coupon_fifteen_used_count'] = (
        (df.loc[day_fifteen, ['date_received', 'date']].count(axis=1) == 2)
        .groupby(df['coupon_id']).sum()
    )

    # 优惠券使用率
    coupon_consume_times = (
        df.loc[day_fifteen, 'date':'date'].count(axis=1)
        .groupby(df['coupon_id']).sum()
    )
    data_coupon['coupon_used_rate'] = (
        data_coupon['coupon_fifteen_used_count'] / coupon_consume_times
    )
    data_coupon['coupon_used_rate'] = data_coupon['coupon_used_rate'].fillna(0)

    return data_coupon


def merge_features(df, data_user, data_merchant, data_coupon):
    """将三组特征与原始数据合并（只对特征列填0，不动日期列）"""
    if 'coupon_id' not in data_coupon.columns:
        data_coupon = data_coupon.reset_index().rename(columns={'index': 'coupon_id'})

    df_merged = pd.merge(df, data_user, on='user_id', how='left')
    df_merged = pd.merge(df_merged, data_merchant, on='merchant_id', how='left')
    df_merged = pd.merge(df_merged, data_coupon, on='coupon_id', how='left')

    # ★ 关键：只对特征列填充 0，绝不碰 date / date_received
    feature_cols = (
        [c for c in data_user.columns if c != 'user_id'] +
        [c for c in data_merchant.columns if c != 'merchant_id'] +
        [c for c in data_coupon.columns if c != 'coupon_id']
    )
    for col in feature_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)

    return df_merged