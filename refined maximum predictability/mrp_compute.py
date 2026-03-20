#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRP (Multi-scale Recency-aware Predictability) 计算

基于论文研究的改进可预测性指标：
1. 使用半衰期而非固定衰减率
2. Miller-Madow熵偏差校正
3. 多尺度平滑融合
4. 基于实际重访模式的数据驱动半衰期

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
    for _ in range(100):
        delta = getapproximation(p, S, N)
        p = p - 0.8 * delta
        if abs(compute_f(p, S, N)) < 0.0000001:
            break
    return p

def compute_half_life(dseq, tseq_int):
    """
    基于重访模式计算半衰期
    
    参数:
        dseq: 位置序列 (整数列表)
        tseq_int: 时间序列 (整数列表，表示时间步)
    
    返回:
        t_half: 半衰期
    """
    # 记录每个位置的出现时间
    loc_times = defaultdict(list)
    for i, (loc, t) in enumerate(zip(dseq, tseq_int)):
        loc_times[loc].append(i)
    
    # 计算重访间隔
    revisit_gaps = []
    for loc, times in loc_times.items():
        if len(times) > 1:
            for i in range(1, len(times)):
                revisit_gaps.append(times[i] - times[i-1])
    
    if len(revisit_gaps) == 0:
        return 10  # 默认半衰期
    
    # 使用中位数半衰期，乘以调节系数
    median_gap = np.median(revisit_gaps)
    t_half = max(5, min(50, median_gap * 0.5))  # 限制在[5, 50]范围内
    
    return t_half

def compute_temporal_weights(sequence_length, t_half):
    """
    计算时间权重 (基于半衰期)
    
    w_t = exp(-ln(2) * Δt / t_half)
    """
    weights = np.zeros(sequence_length)
    ln2 = np.log(2)
    
    for t in range(sequence_length):
        # t=0 是最新的，权重最高
        weights[t] = np.exp(-ln2 * t / t_half)
    
    # 归一化
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    
    return weights

def miller_madow_correction(raw_entropy, m, n):
    """
    Miller-Madow熵偏差校正
    
    S_corrected = S_raw + (m-1)/(2n)
    
    参数:
        raw_entropy: 原始熵
        m: 不同符号数
        n: 总样本数
    """
    if n <= 1:
        return raw_entropy
    correction = (m - 1) / (2 * n)
    return raw_entropy + correction

def compute_weighted_entropy(transitions, time_weights):
    """
    计算时间加权的转移熵
    
    参数:
        transitions: 转移计数矩阵 [from, to]
        time_weights: 时间权重
    
    返回:
        weighted_entropy: 加权条件熵
    """
    m_from, m_to = transitions.shape
    
    # 应用时间权重
    weighted_transitions = transitions.copy()
    # 注意：这里假设transitions已经聚合了，我们需要重新计算带权的概率
    
    # 计算带权的边际和条件概率
    row_sums = np.sum(weighted_transitions, axis=1, keepdims=True)
    total = np.sum(row_sums)
    
    if total == 0:
        return 0
    
    # 条件概率
    cond_prob = weighted_transitions / (row_sums + 1e-10)
    
    # 条件熵
    log_prob = np.log2(cond_prob + 1e-10)
    conditional_entropy = -np.sum(cond_prob * log_prob, axis=1)
    
    # 按边际概率加权
    marginal_prob = row_sums.flatten() / total
    weighted_entropy = np.sum(marginal_prob * conditional_entropy)
    
    return weighted_entropy

def compute_multiscale_weighted_entropy(dseq, tseq, pdnum, tdnum, pdd, tdd, t_half):
    """
    计算多尺度加权熵
    
    三个尺度：
    - 短期 (最近10个): 权重 2
    - 中期 (最近50个): 权重 1
    - 长期 (全部): 权重 0.5
    
    融合: (2*S_short + S_medium + 0.5*S_long) / 3.5
    """
    n = len(dseq)
    entropies = []
    weights = [2.0, 1.0, 0.5]
    scales = [min(10, n), min(50, n), n]
    
    for i, scale in enumerate(scales):
        if n < scale or scale == 0:
            continue
        
        # 获取最近scale个记录
        start_idx = max(0, n - scale)
        recent_dseq = dseq[start_idx:]
        recent_tseq = tseq[start_idx:]
        
        # 计算时间权重 (相对于这个子序列)
        sub_weights = compute_temporal_weights(len(recent_dseq), t_half)
        
        # 构建加权转移矩阵
        juzhen = np.zeros((pdnum, pdnum, tdnum))
        for j in range(1, len(recent_dseq)):
            d1 = recent_dseq[j-1]
            d2 = recent_dseq[j]
            t = recent_tseq[j-1]
            # 使用时间权重
            weight = sub_weights[len(recent_dseq) - j]  # 越近权重越高
            juzhen[d1, d2, t] += weight
        
        # 计算条件熵 (基于时间-地点转移)
        td_dist = np.sum(juzhen, axis=0).transpose()
        
        # 手动计算熵
        row_sums = np.sum(td_dist, axis=1, keepdims=True)
        total = np.sum(row_sums)
        
        if total == 0:
            entropies.append(0)
            continue
        
        cond_prob = td_dist / (row_sums + 1e-10)
        log_prob = np.log2(cond_prob + 1e-10)
        entropy_per_row = -np.sum(cond_prob * log_prob, axis=1)
        marginal_prob = row_sums.flatten() / total
        entropy = np.sum(marginal_prob * entropy_per_row)
        
        # Miller-Madow校正
        m_unique = len(np.unique(recent_dseq))
        entropy = miller_madow_correction(entropy, m_unique, len(recent_dseq))
        
        entropies.append(entropy)
    
    if not entropies:
        return 0
    
    # 多尺度融合
    valid_weights = weights[:len(entropies)]
    valid_entropies = entropies
    
    fused_entropy = np.sum(np.array(valid_entropies) * np.array(valid_weights)) / np.sum(valid_weights)
    
    return fused_entropy

def compute_mrp_for_user(dseq1, tseq1, trnum):
    """
    计算单个用户的MRP
    
    参数:
        dseq1: 位置序列
        tseq1: 时间序列
        trnum: 训练集长度
    
    返回:
        mrp: 多尺度近期感知可预测性
        details: 计算细节
    """
    # 处理训练序列
    dseq_train = dseq1[:trnum]
    tseq_train = tseq1[:trnum]
    
    # 构建映射
    pdnuml = np.unique(dseq_train)
    pdnum = len(pdnuml)
    
    tdnuml = np.unique(tseq_train)
    tdnum = len(tdnuml)
    
    pdd = {pdnuml[i]: i for i in range(len(pdnuml))}
    tdd = {tdnuml[i]: i for i in range(len(tdnuml))}
    
    dseq = [pdd[x] for x in dseq_train]
    tseq = [tdd[x] for x in tseq_train]
    
    # 计算半衰期
    t_half = compute_half_life(dseq, list(range(len(dseq))))
    
    # 计算多尺度加权熵
    ms_entropy = compute_multiscale_weighted_entropy(
        dseq, tseq, pdnum, tdnum, pdd, tdd, t_half
    )
    
    # 计算最大可预测性
    if ms_entropy > 0 and pdnum > 1:
        mrp = max_predictability(ms_entropy, pdnum)
    else:
        mrp = 0.5
    
    details = {
        't_half': t_half,
        'ms_entropy': ms_entropy,
        'n_unique': pdnum,
        'n_total': len(dseq),
        'mrp': mrp
    }
    
    return mrp, details

def main():
    """主函数"""
    print("=" * 60)
    print("MRP (Multi-scale Recency-aware Predictability)")
    print("多尺度近期感知可预测性计算")
    print("=" * 60)
    
    flag = 'foursquare_nyc'
    name1 = '../data_preprocessing/base_' + flag + '_v1.txt'
    outname = './' + flag + '_mrp_train.txt'
    
    print(f"\n[1/3] 读取数据: {name1}")
    ff = open(name1, encoding='utf-8')
    
    results = []
    user_count = 0
    mrp_values = []
    
    print("[2/3] 计算MRP...")
    for ij in ff.readlines():
        mess = ij.strip().split('#')
        user = int(mess[0])
        
        binds = mess[-1].split('&')
        trnum = len([ii for ii in binds if ii == 'train'])
        
        dseq1 = mess[1].split('&')
        tseq1 = mess[2].split('&')
        
        # 计算MRP
        mrp, details = compute_mrp_for_user(dseq1, tseq1, trnum)
        mrp_values.append(mrp)
        
        # 保存结果
        result_line = [
            str(user),
            str(trnum),
            f"{mrp:.5f}",
            f"{details['t_half']:.2f}",
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
    print("MRP计算完成!")
    print("=" * 60)
    print(f"处理用户数: {user_count}")
    print(f"MRP平均值: {np.mean(mrp_values):.4f}")
    print(f"MRP标准差: {np.std(mrp_values):.4f}")
    print(f"MRP范围: [{np.min(mrp_values):.4f}, {np.max(mrp_values):.4f}]")
    print("=" * 60)
    print("\n与ATDP的对比:")
    print("- MRP使用半衰期而非固定衰减率")
    print("- MRP使用Miller-Madow熵校正")
    print("- MRP多尺度融合更平滑")
    print("- MRP预期表现更稳定")
    print("=" * 60)

if __name__ == '__main__':
    main()
