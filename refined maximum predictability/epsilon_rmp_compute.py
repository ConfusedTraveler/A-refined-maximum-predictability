#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ε-RMP (Epsilon-Robust Multi-scale Predictability) 
容差鲁棒多尺度可预测性指标

基于问题分析和论文改进：
1. 容差匹配（来自Jamal Mohammed论文）
2. 软融合（解决硬分类问题）
3. 自适应容差（数据驱动）
4. 平衡多尺度（解决权重问题）
5. 改进的Pmax公式

Author: AI Assistant
Date: 2026-03-20
"""

import numpy as np
import pandas as pd
from collections import defaultdict

def compute_pmax_jamal(entropy_rate, N, max_iter=100, tol=1e-10):
    """
    使用Jamal Mohammed论文的公式求解Pmax
    
    H = Pmax*log2(Pmax) + (1-Pmax)*[log2(1-Pmax) + log2(N-2)]
    
    参数:
        entropy_rate: 熵率
        N: 有效字母表大小
    
    返回:
        Pmax: 最大可预测性
    """
    if entropy_rate <= 0:
        return 0.999
    if N <= 2:
        return 0.999
    
    # 牛顿迭代求解
    p = 0.7  # 初始值
    
    for _ in range(max_iter):
        if p <= 0 or p >= 1:
            p = 0.5
            
        # 计算函数值
        log2_p = np.log2(p) if p > 0 else -100
        log2_1p = np.log2(1-p) if (1-p) > 0 else -100
        log2_N = np.log2(N-2) if (N-2) > 0 else 0
        
        f = entropy_rate - (p * log2_p + (1-p) * (log2_1p + log2_N))
        
        if abs(f) < tol:
            break
        
        # 计算导数
        df = -(log2_p - log2_1p - log2_N)
        
        if abs(df) < 1e-10:
            break
        
        # 更新
        p_new = p - f / df
        p_new = np.clip(p_new, 0.001, 0.999)
        
        if abs(p_new - p) < tol:
            break
        
        p = p_new
    
    return p

def tolerance_match(val1, val2, epsilon):
    """
    容差匹配（核心改进1：来自Jamal Mohammed论文）
    
    参数:
        val1, val2: 待匹配的值
        epsilon: 容差阈值
    
    返回:
        bool: 是否在容差范围内匹配
    """
    return abs(float(val1) - float(val2)) <= epsilon

def compute_adaptive_epsilon(sequence, alpha=0.1):
    """
    计算自适应容差（核心改进2：数据驱动）
    
    epsilon = alpha * std(sequence)
    
    参数:
        sequence: 数值序列
        alpha: 调节参数（默认0.1）
    
    返回:
        epsilon: 容差值
    """
    if len(sequence) < 2:
        return 1.0
    
    std_val = np.std(sequence)
    epsilon = max(1.0, alpha * std_val)  # 至少为1.0
    
    return epsilon

def compute_tolerance_entropy_rate(dseq, tseq, epsilon):
    """
    计算容差感知的熵率（核心改进3：容差感知）
    
    使用容差匹配统计转移模式，而非精确匹配
    
    参数:
        dseq: 位置序列（整数列表）
        tseq: 时间序列
        epsilon: 容差阈值
    
    返回:
        entropy_rate: 容差感知的熵率
    """
    n = len(dseq)
    if n < 2:
        return 0
    
    # 统计容差感知的转移
    transition_counts = defaultdict(lambda: defaultdict(int))
    
    for i in range(1, n):
        prev_loc = dseq[i-1]
        curr_loc = dseq[i]
        prev_time = tseq[i-1]
        
        # 使用容差匹配找历史中的相似转移
        for j in range(i):
            if tolerance_match(dseq[j], prev_loc, epsilon) and \
               tseq[j] == prev_time:
                # 找到相似的历史状态
                next_loc = dseq[j+1] if j+1 < len(dseq) else curr_loc
                transition_counts[(prev_loc, prev_time)][next_loc] += 1
    
    if not transition_counts:
        return np.log2(len(set(dseq)))
    
    # 计算条件熵
    total_entropy = 0
    total_count = 0
    
    for (loc, time), targets in transition_counts.items():
        count_sum = sum(targets.values())
        if count_sum == 0:
            continue
        
        # 条件概率分布
        probs = [c / count_sum for c in targets.values()]
        
        # 熵
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        total_entropy += entropy * count_sum
        total_count += count_sum
    
    if total_count == 0:
        return 0
    
    entropy_rate = total_entropy / total_count
    
    # Miller-Madow校正
    m = len(set(dseq))
    correction = (m - 1) / (2 * n)
    entropy_rate = min(entropy_rate + correction, np.log2(m))
    
    return entropy_rate

def compute_multiscale_tolerance_entropy(dseq, tseq, base_epsilon):
    """
    多尺度容差熵（核心改进4：容差金字塔）
    
    不同时间尺度使用不同容差：
    - 短期：小容差（精确匹配）
    - 中期：中等容差
    - 长期：大容差（粗粒度）
    
    参数:
        dseq: 位置序列
        tseq: 时间序列
        base_epsilon: 基础容差
    
    返回:
        fused_entropy: 融合后的熵
        entropies: 各尺度熵列表
    """
    n = len(dseq)
    
    # 多尺度容差设置
    scales = [
        (min(10, n), base_epsilon * 0.5),      # 短期：小容差
        (min(30, n), base_epsilon * 1.0),      # 中期：标准容差
        (n, base_epsilon * 2.0)                 # 长期：大容差
    ]
    
    entropies = []
    weights = [1.2, 1.0, 0.8]  # 平衡权重（解决MRP的过度短期问题）
    
    for (scale_len, epsilon), weight in zip(scales, weights):
        if n < scale_len:
            continue
        
        # 截取最近scale_len个记录
        start_idx = max(0, n - scale_len)
        sub_dseq = dseq[start_idx:]
        sub_tseq = tseq[start_idx:]
        
        # 计算该尺度的容差熵
        entropy = compute_tolerance_entropy_rate(sub_dseq, sub_tseq, epsilon)
        entropies.append((entropy, weight))
    
    if not entropies:
        return 0, []
    
    # 加权融合
    weighted_sum = sum(e * w for e, w in entropies)
    weight_sum = sum(w for _, w in entropies)
    
    fused_entropy = weighted_sum / weight_sum
    
    return fused_entropy, [e for e, _ in entropies]

def compute_epsilon_rmp_for_user(dseq1, tseq1, trnum):
    """
    计算单个用户的ε-RMP
    
    参数:
        dseq1: 位置序列（字符串列表）
        tseq1: 时间序列（字符串列表）
        trnum: 训练集长度
    
    返回:
        epsilon_rmp: ε-RMP值
        details: 计算细节
    """
    # 处理训练序列
    dseq_train = dseq1[:trnum]
    tseq_train = tseq1[:trnum]
    
    # 转换为整数
    try:
        dseq_int = [int(x) for x in dseq_train]
        # 时间编码（简化处理）
        tseq_int = []
        for t in tseq_train:
            parts = t.split('_')
            if len(parts) >= 4:
                # 编码为小时（0-47，区分工作日/周末）
                hour = int(parts[3])
                weekday = int(parts[2])
                if weekday > 4:
                    hour += 24
                tseq_int.append(hour)
            else:
                tseq_int.append(0)
    except:
        return 0.5, {'error': 'conversion failed'}
    
    # 计算自适应容差
    epsilon = compute_adaptive_epsilon(dseq_int, alpha=0.15)
    
    # 计算多尺度容差熵
    ms_entropy, entropies = compute_multiscale_tolerance_entropy(
        dseq_int, tseq_int, epsilon
    )
    
    # 计算有效字母表大小
    unique_vals = set(dseq_int)
    N_eff = max(len(unique_vals), int((max(dseq_int) - min(dseq_int)) / epsilon) + 2)
    
    # 计算ε-RMP（使用Jamal Mohammed的公式）
    if ms_entropy > 0 and N_eff > 2:
        epsilon_rmp = compute_pmax_jamal(ms_entropy, N_eff)
    else:
        epsilon_rmp = 0.5
    
    details = {
        'epsilon': epsilon,
        'ms_entropy': ms_entropy,
        'entropies': entropies,
        'N_eff': N_eff,
        'n_unique': len(unique_vals),
        'n_total': len(dseq_int),
        'epsilon_rmp': epsilon_rmp
    }
    
    return epsilon_rmp, details

def main():
    """主函数"""
    print("=" * 60)
    print("ε-RMP (Epsilon-Robust Multi-scale Predictability)")
    print("容差鲁棒多尺度可预测性指标")
    print("=" * 60)
    
    flag = 'foursquare_nyc'
    name1 = '../data_preprocessing/base_' + flag + '_v1.txt'
    outname = './' + flag + '_epsilon_rmp_train.txt'
    
    print(f"\n[1/3] 读取数据: {name1}")
    ff = open(name1, encoding='utf-8')
    
    results = []
    user_count = 0
    epsilon_rmp_values = []
    
    print("[2/3] 计算ε-RMP...")
    for ij in ff.readlines():
        mess = ij.strip().split('#')
        user = int(mess[0])
        
        binds = mess[-1].split('&')
        trnum = len([ii for ii in binds if ii == 'train'])
        
        dseq1 = mess[1].split('&')
        tseq1 = mess[2].split('&')
        
        # 计算ε-RMP
        epsilon_rmp, details = compute_epsilon_rmp_for_user(dseq1, tseq1, trnum)
        epsilon_rmp_values.append(epsilon_rmp)
        
        # 保存结果
        result_line = [
            str(user),
            str(trnum),
            f"{epsilon_rmp:.5f}",
            f"{details.get('epsilon', 0):.2f}",
            f"{details.get('ms_entropy', 0):.5f}",
            str(details.get('n_unique', 0)),
            str(details.get('n_total', 0))
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
    print("ε-RMP计算完成!")
    print("=" * 60)
    print(f"处理用户数: {user_count}")
    print(f"ε-RMP平均值: {np.mean(epsilon_rmp_values):.4f}")
    print(f"ε-RMP标准差: {np.std(epsilon_rmp_values):.4f}")
    print(f"ε-RMP范围: [{np.min(epsilon_rmp_values):.4f}, {np.max(epsilon_rmp_values):.4f}]")
    print("=" * 60)
    print("\n核心改进:")
    print("1. 容差匹配（Jamal Mohammed论文）")
    print("2. 自适应容差（数据驱动）")
    print("3. 多尺度容差金字塔")
    print("4. 平衡权重（解决MRP问题）")
    print("5. 改进Pmax公式")
    print("=" * 60)

if __name__ == '__main__':
    main()
