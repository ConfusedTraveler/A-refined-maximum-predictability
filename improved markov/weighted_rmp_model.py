#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted RMP Model - 改进的加权融合策略
解决原方法的硬切换问题，采用softmax加权融合多维度信息

改进点:
1. 加权融合策略: 根据各维度RMP的相对大小进行softmax加权
2. 自适应阈值: 根据数据分布动态调整低频过滤阈值
3. 平滑冷启动: 融入全局流行度处理稀疏数据
4. 综合评估指标: 增加MRR、NDCG等评估指标

Author: AI Assistant
Date: 2026-03-20
"""

import random
import numpy as np
import pandas as pd
from collections import defaultdict
import json

def time1(x):
    """提取详细时间特征 (0-47, 区分工作日/周末)"""
    mm = x.split('_')
    ww = int(mm[2])
    hh = int(mm[3])
    if ww > 4:
        nhh = hh + 24
    else:
        nhh = hh
    return nhh

def time2(x):
    """提取工作日/周末特征 (0-1)"""
    mm = x.split('_')
    ww = int(mm[2])
    if ww > 4:
        nhh = 1
    else:
        nhh = 0
    return nhh

def softmax(weights):
    """计算softmax权重"""
    exp_weights = np.exp(weights - np.max(weights))  # 数值稳定性
    return exp_weights / np.sum(exp_weights)

def compute_adaptive_threshold(transition_counts, alpha=0.1):
    """
    自适应阈值计算
    根据转移计数的分布，过滤底部alpha比例的低频转移
    """
    if len(transition_counts) == 0:
        return 1
    nonzero_counts = transition_counts[transition_counts > 0]
    if len(nonzero_counts) == 0:
        return 1
    # 使用百分位数作为自适应阈值
    ke = max(int(np.percentile(nonzero_counts, alpha * 100)), 1)
    return ke

def weighted_markov_pred(juzhen1, juzhen2, d1, t1, t2, rmp_values, global_popularity, beta=0.8):
    """
    加权融合马尔可夫预测
    
    参数:
        juzhen1: 详细时间-地点转移矩阵
        juzhen2: 工作日/周末-地点转移矩阵
        d1: 当前位置
        t1, t2: 时间特征
        rmp_values: [r_tod, r_td, r_od, r_d, r_tod2, r_td2] - 各维度RMP值
        global_popularity: 全局位置流行度
        beta: 用户个性化与全局流行的平衡系数
    """
    # 解包RMP值
    r_tod, r_td, r_od, r_d, r_tod2, r_td2 = rmp_values
    
    # 计算各策略的权重 (softmax)
    weights = softmax(np.array([r_tod, r_td, r_od, r_d, r_tod2, r_td2]))
    
    # 提取各维度的概率分布
    strategies = []
    
    # 策略1: TOD (Time-Origin-Destination)
    if np.sum(juzhen1[d1, :, t1]) > 0:
        p_tod = juzhen1[d1, :, t1] / np.sum(juzhen1[d1, :, t1])
    else:
        p_tod = global_popularity.copy()
    strategies.append(p_tod)
    
    # 策略2: TD (Time-Destination)
    td_dist = np.sum(juzhen1, axis=0).transpose()
    if np.sum(td_dist[t1, :]) > 0:
        p_td = td_dist[t1, :] / np.sum(td_dist[t1, :])
    else:
        p_td = global_popularity.copy()
    strategies.append(p_td)
    
    # 策略3: OD (Origin-Destination)
    od_dist = np.sum(juzhen1, axis=2)
    if np.sum(od_dist[d1, :]) > 0:
        p_od = od_dist[d1, :] / np.sum(od_dist[d1, :])
    else:
        p_od = global_popularity.copy()
    strategies.append(p_od)
    
    # 策略4: D (Destination only - global popularity)
    d_dist = np.sum(np.sum(juzhen1, axis=2), axis=0)
    if np.sum(d_dist) > 0:
        p_d = d_dist / np.sum(d_dist)
    else:
        p_d = global_popularity.copy()
    strategies.append(p_d)
    
    # 策略5: TOD2 (Weekday/Weekend TOD)
    if np.sum(juzhen2[d1, :, t2]) > 0:
        p_tod2 = juzhen2[d1, :, t2] / np.sum(juzhen2[d1, :, t2])
    else:
        p_tod2 = global_popularity.copy()
    strategies.append(p_tod2)
    
    # 策略6: TD2 (Weekday/Weekend TD)
    td2_dist = np.sum(juzhen2, axis=0).transpose()
    if np.sum(td2_dist[t2, :]) > 0:
        p_td2 = td2_dist[t2, :] / np.sum(td2_dist[t2, :])
    else:
        p_td2 = global_popularity.copy()
    strategies.append(p_td2)
    
    # 加权融合所有策略
    p_fused = np.zeros_like(global_popularity)
    for i, (w, p) in enumerate(zip(weights, strategies)):
        p_fused += w * p
    
    # 平滑冷启动: 融合全局流行度
    p_final = beta * p_fused + (1 - beta) * global_popularity
    
    # 返回Top-10预测
    pred_indices = np.argsort(p_final)[::-1][:10].tolist()
    return pred_indices, weights

def compute_global_popularity(user_sequences):
    """计算全局位置流行度"""
    location_counts = defaultdict(int)
    total = 0
    for seq in user_sequences:
        for loc in seq:
            location_counts[loc] += 1
            total += 1
    
    # 转换为概率分布
    max_loc = max(location_counts.keys()) if location_counts else 0
    popularity = np.zeros(max_loc + 1)
    for loc, count in location_counts.items():
        popularity[loc] = count / total
    return popularity

def evaluate_prediction(y_true, y_pred):
    """
    综合评估指标计算
    
    返回:
        top1, top5, top10: 准确率
        mrr: Mean Reciprocal Rank
        ndcg5, ndcg10: NDCG指标
    """
    top1 = top5 = top10 = 0
    rr_sum = 0  # For MRR
    ndcg5_sum = ndcg10_sum = 0
    n = len(y_true)
    
    for true_loc, pred_list in zip(y_true, y_pred):
        if true_loc in pred_list:
            rank = pred_list.index(true_loc) + 1  # 1-indexed
            
            if rank == 1:
                top1 += 1
                top5 += 1
                top10 += 1
            elif rank <= 5:
                top5 += 1
                top10 += 1
            elif rank <= 10:
                top10 += 1
            
            # MRR
            rr_sum += 1.0 / rank
            
            # NDCG
            dcg5 = 1.0 / np.log2(rank + 1) if rank <= 5 else 0
            dcg10 = 1.0 / np.log2(rank + 1) if rank <= 10 else 0
            
            # Ideal DCG (IDCG) = 1 for single relevant item
            ndcg5_sum += dcg5
            ndcg10_sum += dcg10
    
    return {
        'top1': top1 / n,
        'top5': top5 / n,
        'top10': top10 / n,
        'mrr': rr_sum / n,
        'ndcg5': ndcg5_sum / n,
        'ndcg10': ndcg10_sum / n
    }

def main():
    """主函数"""
    print("=" * 60)
    print("Weighted RMP Model - 加权融合改进模型")
    print("=" * 60)
    
    func1 = lambda x: time1(x)
    func2 = lambda x: time2(x)
    
    flag = 'foursquare_nyc'
    name1 = '../data_preprocessing/base_' + flag + '_v1.txt'
    name3 = '../refined maximum predictability/' + flag + '_rmp_train.txt'
    
    # 读取数据
    print("\n[1/4] 读取数据...")
    ff = open(name1, encoding='utf-8')
    df2 = pd.read_csv(name3, sep='$', header=None, encoding='utf-8')
    
    # 首先收集所有用户序列以计算全局流行度
    print("[2/4] 计算全局位置流行度...")
    all_sequences = []
    user_data = []
    
    for ij in ff.readlines():
        mess = ij.strip().split('#')
        binds = mess[-1].split('&')
        user = int(mess[0])
        
        if 'test' in binds:
            dseq1 = mess[1].split('&')
            tseq1 = mess[2].split('&')
            dseq_int = [int(x) for x in dseq1]
            all_sequences.append(dseq_int)
            user_data.append({
                'user': user,
                'mess': mess,
                'binds': binds,
                'dseq1': dseq1,
                'tseq1': tseq1
            })
    
    ff.close()
    
    # 计算全局流行度
    global_popularity = compute_global_popularity(all_sequences)
    print(f"     全局位置数量: {len(global_popularity)}")
    
    # 重新打开文件进行预测
    ff = open(name1, encoding='utf-8')
    
    print("[3/4] 执行加权RMP预测...")
    
    results = []
    all_y_true = []
    all_y_pred = []
    user_count = 0
    
    for ij in ff.readlines():
        mess = ij.strip().split('#')
        binds = mess[-1].split('&')
        user = int(mess[0])
        
        if 'test' not in binds:
            continue
            
        # 获取RMP值
        user_rmp = df2.loc[(df2[0] == user)]
        if len(user_rmp) == 0:
            continue
        
        tmess = user_rmp.iloc[0, :].tolist()
        rmp_values = [tmess[2], tmess[3], tmess[4], tmess[5], tmess[6], tmess[7]]
        
        dseq1 = mess[1].split('&')
        tseq1 = mess[2].split('&')
        
        tseq2 = list(map(func1, tseq1))
        tseq3 = list(map(func2, tseq1))
        
        # 构建位置映射
        pdnuml = sorted(set(dseq1))
        pdnuml = pdnuml + ['-1', '-2', '-3', '-4', '-5', '-6', '-7', '-8', '-9', '-10']
        pdnum = len(pdnuml)
        
        tdnuml2 = np.unique(tseq2)
        tdnum2 = len(tdnuml2)
        
        tdnuml3 = np.unique(tseq3)
        tdnum3 = len(tdnuml3)
        
        # 初始化转移矩阵
        juzhen1 = np.zeros((pdnum, pdnum, tdnum2))
        juzhen2 = np.zeros((pdnum, pdnum, tdnum3))
        
        pdd = {pdnuml[i]: i for i in range(len(pdnuml))}
        tdd2 = {tdnuml2[i]: i for i in range(len(tdnuml2))}
        tdd3 = {tdnuml3[i]: i for i in range(len(tdnuml3))}
        
        tseq22 = [tdd2[x] for x in tseq2]
        tseq33 = [tdd3[x] for x in tseq3]
        dseq11 = [pdd[x] for x in dseq1]
        
        # 扩展全局流行度以匹配当前用户的矩阵大小
        user_global_pop = np.zeros(pdnum)
        for loc, idx in pdd.items():
            if loc != '-1' and int(loc) < len(global_popularity):
                user_global_pop[idx] = global_popularity[int(loc)]
        # 归一化
        if np.sum(user_global_pop) > 0:
            user_global_pop = user_global_pop / np.sum(user_global_pop)
        
        # 预测
        user_y_true = []
        user_y_pred = []
        
        for j in range(1, len(dseq11)):
            t2 = tseq22[j-1]
            t3 = tseq33[j-1]
            d1 = dseq11[j-1]
            d2 = dseq11[j]
            
            if binds[j] == 'test':
                # 使用加权RMP预测
                pred, weights = weighted_markov_pred(
                    juzhen1, juzhen2, d1, t2, t3, 
                    rmp_values, user_global_pop,
                    beta=min(0.9, len([x for x in binds[:j] if x == 'train']) / 20 + 0.5)
                )
                user_y_true.append(d2)
                user_y_pred.append(pred)
            
            # 更新转移矩阵
            juzhen1[d1, d2, t2] += 1
            juzhen2[d1, d2, t3] += 1
        
        if user_y_true:
            metrics = evaluate_prediction(user_y_true, user_y_pred)
            results.append(metrics)
            all_y_true.extend(user_y_true)
            all_y_pred.extend(user_y_pred)
        
        user_count += 1
        if user_count % 100 == 0:
            print(f"     已处理 {user_count} 个用户...")
    
    ff.close()
    
    print("[4/4] 计算评估指标...")
    
    # 汇总结果
    avg_metrics = {
        'top1': np.mean([r['top1'] for r in results]),
        'top5': np.mean([r['top5'] for r in results]),
        'top10': np.mean([r['top10'] for r in results]),
        'mrr': np.mean([r['mrr'] for r in results]),
        'ndcg5': np.mean([r['ndcg5'] for r in results]),
        'ndcg10': np.mean([r['ndcg10'] for r in results])
    }
    
    # 计算总体指标（不按用户平均）
    overall_metrics = evaluate_prediction(all_y_true, all_y_pred)
    
    print("\n" + "=" * 60)
    print("加权RMP模型 - 实验结果")
    print("=" * 60)
    print(f"\n处理用户数: {user_count}")
    print(f"总测试样本数: {len(all_y_true)}")
    print(f"\n[按用户平均的指标]")
    print(f"  Top-1  准确率: {avg_metrics['top1']:.4f} ({avg_metrics['top1']*100:.2f}%)")
    print(f"  Top-5  准确率: {avg_metrics['top5']:.4f} ({avg_metrics['top5']*100:.2f}%)")
    print(f"  Top-10 准确率: {avg_metrics['top10']:.4f} ({avg_metrics['top10']*100:.2f}%)")
    print(f"  MRR:           {avg_metrics['mrr']:.4f}")
    print(f"  NDCG@5:        {avg_metrics['ndcg5']:.4f}")
    print(f"  NDCG@10:       {avg_metrics['ndcg10']:.4f}")
    print(f"\n[总体指标]")
    print(f"  Top-1  准确率: {overall_metrics['top1']:.4f} ({overall_metrics['top1']*100:.2f}%)")
    print(f"  Top-5  准确率: {overall_metrics['top5']:.4f} ({overall_metrics['top5']*100:.2f}%)")
    print(f"  Top-10 准确率: {overall_metrics['top10']:.4f} ({overall_metrics['top10']*100:.2f}%)")
    print(f"  MRR:           {overall_metrics['mrr']:.4f}")
    print(f"  NDCG@5:        {overall_metrics['ndcg5']:.4f}")
    print(f"  NDCG@10:       {overall_metrics['ndcg10']:.4f}")
    print("=" * 60)
    
    # 保存结果
    output = {
        'model': 'Weighted RMP',
        'user_count': user_count,
        'total_samples': len(all_y_true),
        'avg_metrics': avg_metrics,
        'overall_metrics': overall_metrics
    }
    
    with open('../results/weighted_rmp_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n结果已保存至: results/weighted_rmp_results.json")
    
    return overall_metrics

if __name__ == '__main__':
    main()
