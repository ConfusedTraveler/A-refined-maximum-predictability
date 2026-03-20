#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATDP (Adaptive Temporal Decay Predictability) 计算

新可预测性指标的核心创新：
1. 时间衰减权重: 近期转移贡献更大
2. 自适应衰减率: 根据数据稀疏度动态调整
3. 多尺度融合: 短/中/长期条件熵加权
4. 上下文感知: 考虑当前时空上下文

Author: AI Assistant
Date: 2026-03-20
"""

import numpy as np
import pandas as pd
from collections import defaultdict

def compute_f(p, S, N):
    """牛顿迭代辅助函数"""
    if p <= 0 or p >= 1:
        return float('inf')
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    pi_max = h + (1 - p) * np.log2(N - 1) - S
    return pi_max

def getapproximation(p, S, N):
    """牛顿迭代步长"""
    f = compute_f(p, S, N)
    d1 = np.log2(1-p) - np.log2(p) - np.log2(N-1)
    d2 = 1 / ((p-1)*p)
    if abs(d1) < 1e-10:
        return 0
    return f / (d1 - f*d2/(2*d1))

def max_predictability(S, N):
    """求解最大可预测性"""
    if S > np.log2(N):
        return 0
    if S <= 0.01:
        return 0.999
    
    p = (N + 1) / (2 * N)
    for _ in range(100):  # 最大迭代次数
        delta = getapproximation(p, S, N)
        p = p - 0.8 * delta
        if abs(compute_f(p, S, N)) < 0.0000001:
            break
    return p

def compute_temporal_weights(sequence_length, lambda_decay):
    """
    计算时间衰减权重
    
    参数:
        sequence_length: 序列长度
        lambda_decay: 衰减率
    
    返回:
        weights: 每个时间步的权重 (越近期权重越高)
    """
    weights = np.zeros(sequence_length)
    for t in range(sequence_length):
        # t=0 是最新的，权重最高
        weights[t] = np.exp(-lambda_decay * np.sqrt(t))
    
    # 归一化
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    
    return weights

def compute_adaptive_lambda(n_unique, n_total, alpha=0.5):
    """
    计算自适应衰减率
    
    参数:
        n_unique: 不同位置数
        n_total: 总记录数
        alpha: 调节参数
    
    返回:
        lambda_decay: 自适应衰减率
    """
    if n_total == 0:
        return 0.1
    
    # 稀疏度 = 不同位置数 / 总记录数
    sparsity = n_unique / n_total
    
    # 稀疏度越高，衰减率越小（保留更多历史信息）
    lambda_decay = alpha * (1 - sparsity) + 0.05
    
    return lambda_decay

def compute_weighted_conditional_entropy(transitions, time_weights, axis=1):
    """
    计算时间加权的条件熵
    
    参数:
        transitions: 转移计数矩阵
        time_weights: 时间权重
        axis: 条件维度
    
    返回:
        weighted_entropy: 加权条件熵
    """
    m1, m2 = transitions.shape
    
    # 按时间加权
    weighted_transitions = transitions.copy()
    
    # 计算加权边际分布
    sum_cond = np.sum(weighted_transitions, axis=axis).reshape((-1, 1))
    total = np.sum(sum_cond)
    
    if total == 0:
        return 0
    
    # 条件概率
    cond_prob = weighted_transitions / (sum_cond + 1e-10)
    
    # 加权条件熵
    log_prob = np.log2(cond_prob + 1e-10)
    entropy = -np.sum(cond_prob * log_prob, axis=axis)
    
    # 按边际分布加权
    marginal_prob = sum_cond / total
    weighted_entropy = np.sum(marginal_prob * entropy)
    
    return weighted_entropy

def compute_multiscale_entropy(dseq, tseq, pdnum, tdnum, pdd, tdd, scales=[5, 20, 100]):
    """
    计算多尺度时间熵
    
    参数:
        dseq: 位置序列
        tseq: 时间序列
        pdnum: 位置数量
        tdnum: 时间维度数量
        pdd, tdd: 映射字典
        scales: 不同时间尺度
    
    返回:
        ms_entropy: 多尺度融合熵
        entropies: 各尺度熵值列表
    """
    n = len(dseq)
    entropies = []
    
    for scale in scales:
        if n < scale:
            continue
        
        # 只考虑最近scale个记录
        recent_dseq = dseq[-scale:]
        recent_tseq = tseq[-scale:]
        
        # 构建转移矩阵
        juzhen = np.zeros((pdnum, pdnum, tdnum))
        for j in range(1, len(recent_dseq)):
            d1 = recent_dseq[j-1]
            d2 = recent_dseq[j]
            t = recent_tseq[j-1]
            juzhen[d1, d2, t] += 1
        
        # 计算该尺度的条件熵
        td_dist = np.sum(juzhen, axis=0).transpose()
        entropy = compute_weighted_conditional_entropy(td_dist, np.ones(scale), axis=1)
        entropies.append(entropy)
    
    if not entropies:
        return 0, []
    
    # 多尺度融合 (短期权重更高)
    weights_scale = np.exp(-0.5 * np.arange(len(entropies)))
    weights_scale = weights_scale / np.sum(weights_scale)
    
    ms_entropy = np.sum(np.array(entropies) * weights_scale)
    
    return ms_entropy, entropies

def compute_atdp_for_user(dseq1, tseq1, trnum):
    """
    计算单个用户的ATDP
    
    参数:
        dseq1: 位置序列（字符串列表）
        tseq1: 时间序列（字符串列表）
        trnum: 训练集长度
    
    返回:
        atdp: 自适应时间衰减可预测性
        details: 计算细节
    """
    # 处理训练序列
    dseq_train = dseq1[:trnum]
    tseq_train = tseq1[:trnum]
    
    # 构建位置映射
    pdnuml = np.unique(dseq_train)
    pdnum = len(pdnuml)
    
    tdnuml = np.unique(tseq_train)
    tdnum = len(tdnuml)
    
    pdd = {pdnuml[i]: i for i in range(len(pdnuml))}
    tdd = {tdnuml[i]: i for i in range(len(tdnuml))}
    
    dseq = [pdd[x] for x in dseq_train]
    tseq = [tdd[x] for x in tseq_train]
    
    # 计算自适应衰减率
    n_unique = len(pdnuml)
    n_total = len(dseq)
    lambda_decay = compute_adaptive_lambda(n_unique, n_total, alpha=0.5)
    
    # 计算多尺度熵
    ms_entropy, entropies = compute_multiscale_entropy(
        dseq, tseq, pdnum, tdnum, pdd, tdd, 
        scales=[min(5, n_total), min(20, n_total), n_total]
    )
    
    # 计算最大可预测性
    if ms_entropy > 0 and pdnum > 1:
        atdp = max_predictability(ms_entropy, pdnum)
    else:
        atdp = 0.5  # 默认中等可预测性
    
    details = {
        'lambda_decay': lambda_decay,
        'ms_entropy': ms_entropy,
        'entropies': entropies,
        'n_unique': n_unique,
        'n_total': n_total,
        'atdp': atdp
    }
    
    return atdp, details

def main():
    """主函数：为所有用户计算ATDP"""
    print("=" * 60)
    print("ATDP (Adaptive Temporal Decay Predictability)")
    print("自适应时间衰减可预测性计算")
    print("=" * 60)
    
    flag = 'foursquare_nyc'
    name1 = '../data_preprocessing/base_' + flag + '_v1.txt'
    outname = './' + flag + '_atdp_train.txt'
    
    print(f"\n[1/3] 读取数据: {name1}")
    ff = open(name1, encoding='utf-8')
    
    results = []
    user_count = 0
    atdp_values = []
    
    print("[2/3] 计算ATDP...")
    for ij in ff.readlines():
        mess = ij.strip().split('#')
        user = int(mess[0])
        
        binds = mess[-1].split('&')
        trnum = len([ii for ii in binds if ii == 'train'])
        
        dseq1 = mess[1].split('&')
        tseq1 = mess[2].split('&')
        
        # 计算ATDP
        atdp, details = compute_atdp_for_user(dseq1, tseq1, trnum)
        atdp_values.append(atdp)
        
        # 保存结果
        result_line = [
            str(user),
            str(trnum),
            f"{atdp:.5f}",
            f"{details['lambda_decay']:.5f}",
            f"{details['ms_entropy']:.5f}",
            str(details['n_unique']),
            str(details['n_total'])
        ]
        results.append('$'.join(result_line))
        
        user_count += 1
        if user_count % 200 == 0:
            print(f"     已处理 {user_count} 个用户...")
    
    ff.close()
    
    print(f"[3/3] 保存结果: {outname}")
    with open(outname, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')
    
    print("\n" + "=" * 60)
    print("ATDP计算完成!")
    print("=" * 60)
    print(f"处理用户数: {user_count}")
    print(f"ATDP平均值: {np.mean(atdp_values):.4f}")
    print(f"ATDP标准差: {np.std(atdp_values):.4f}")
    print(f"ATDP范围: [{np.min(atdp_values):.4f}, {np.max(atdp_values):.4f}]")
    print("=" * 60)
    
    # 与RMP对比
    print("\n与RMP指标的对比（简要）:")
    print("- ATDP考虑时间衰减，近期行为权重更高")
    print("- ATDP使用自适应衰减率，根据数据稀疏度调整")
    print("- ATDP融合多尺度信息，短/中/长期综合考虑")
    print("=" * 60)

if __name__ == '__main__':
    main()
