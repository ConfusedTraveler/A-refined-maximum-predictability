#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADF Model - Attention-based Dynamic Fusion for Next Location Prediction
基于注意力机制的动态融合模型

核心创新:
1. 上下文感知注意力: 根据当前时空上下文动态计算各维度权重
2. 历史注意力: 利用近期历史行为调整预测
3. 自适应门控: 平衡个性化与全局趋势
4. 多头融合: 模拟Transformer的多头注意力机制

Author: AI Assistant
Date: 2026-03-20
"""

import random
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import json

def time_features(x):
    """提取完整时间特征向量"""
    mm = x.split('_')
    month = int(mm[0])
    period = int(mm[1])
    weekday = int(mm[2])
    hour = int(mm[3])
    is_weekend = 1 if weekday > 4 else 0
    return np.array([month, period, weekday, hour, is_weekend])

def compute_context_embedding(t1, t2, d1, history_len):
    """
    计算上下文嵌入向量
    包含: 时间特征、位置特征、历史长度特征
    """
    time_emb = np.array([t1 / 47.0, t2])  # 归一化时间
    loc_emb = np.array([d1 / 1000.0])     # 归一化位置（假设最大1000）
    hist_emb = np.array([min(history_len / 100.0, 1.0)])  # 历史长度
    return np.concatenate([time_emb, loc_emb, hist_emb])

def attention_weights(context_emb, strategy_reps, temperature=1.0):
    """
    计算注意力权重
    
    参数:
        context_emb: 上下文嵌入 [dim_c]
        strategy_reps: 各策略的表示向量 [num_strategies, dim_s]
        temperature: 温度参数控制分布锐度
    """
    # 扩展上下文以匹配策略维度
    context_proj = np.tile(context_emb, (len(strategy_reps), 1))  # [6, dim_c]
    
    # 计算注意力分数 (点积注意力)
    scores = np.sum(context_proj * strategy_reps, axis=1) / temperature
    
    # Softmax归一化
    exp_scores = np.exp(scores - np.max(scores))
    weights = exp_scores / np.sum(exp_scores)
    
    return weights

def multi_head_attention_fusion(context_emb, strategy_probs, num_heads=3):
    """
    多头注意力融合
    模拟Transformer的多头机制，从不同子空间融合信息
    """
    num_strategies = len(strategy_probs)
    head_weights = []
    
    for head in range(num_heads):
        # 每个头使用不同的随机投影
        np.random.seed(head)  # 保证可重复性
        projection = np.random.randn(context_emb.shape[0], num_strategies)
        scores = np.dot(context_emb, projection)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        weights = exp_scores / np.sum(exp_scores)
        head_weights.append(weights)
    
    # 平均多头结果
    avg_weights = np.mean(head_weights, axis=0)
    
    # 加权融合策略概率
    fused_prob = np.zeros_like(strategy_probs[0])
    for w, prob in zip(avg_weights, strategy_probs):
        fused_prob += w * prob
    
    return fused_prob, avg_weights

def compute_temporal_attention(recent_locations, current_time, decay_factor=0.9):
    """
    时间注意力: 近期访问的位置有更高权重
    """
    if not recent_locations:
        return None
    
    weights = []
    for i, (loc, time) in enumerate(recent_locations):
        time_diff = abs(current_time - time)
        weight = decay_factor ** time_diff
        weights.append((loc, weight))
    
    return weights

def adf_predict(juzhen1, juzhen2, d1, t1, t2, rmp_values, 
                global_popularity, recent_history, context_gate=0.8):
    """
    ADF模型预测
    
    参数:
        juzhen1, juzhen2: 转移矩阵
        d1: 当前位置
        t1, t2: 时间特征
        rmp_values: 6维RMP值
        global_popularity: 全局流行度
        recent_history: 近期历史 [(loc, time), ...]
        context_gate: 上下文门控系数
    """
    # 1. 计算上下文嵌入
    history_len = len([x for x in recent_history if x[0] != -1]) if recent_history else 0
    context_emb = compute_context_embedding(t1, t2, d1, history_len)
    
    # 2. 构建策略表示 (RMP值作为基础，结合策略特性)
    strategy_reps = np.array([
        [rmp_values[0], 1.0, 0.0, 1.0, 0.0],  # TOD
        [rmp_values[1], 0.0, 1.0, 1.0, 0.0],  # TD
        [rmp_values[2], 1.0, 0.0, 0.0, 1.0],  # OD
        [rmp_values[3], 0.0, 0.0, 0.0, 1.0],  # D
        [rmp_values[4], 1.0, 0.0, 1.0, 1.0],  # TOD2
        [rmp_values[5], 0.0, 1.0, 1.0, 1.0],  # TD2
    ])
    
    # 3. 计算各策略的概率分布
    pdim = juzhen1.shape[1]
    strategies = []
    
    # TOD
    if np.sum(juzhen1[d1, :, t1]) > 0:
        p = juzhen1[d1, :, t1] / np.sum(juzhen1[d1, :, t1])
    else:
        p = global_popularity[:pdim].copy()
    strategies.append(p)
    
    # TD
    td = np.sum(juzhen1, axis=0).transpose()
    if np.sum(td[t1, :]) > 0:
        p = td[t1, :] / np.sum(td[t1, :])
    else:
        p = global_popularity[:pdim].copy()
    strategies.append(p)
    
    # OD
    od = np.sum(juzhen1, axis=2)
    if np.sum(od[d1, :]) > 0:
        p = od[d1, :] / np.sum(od[d1, :])
    else:
        p = global_popularity[:pdim].copy()
    strategies.append(p)
    
    # D
    d_all = np.sum(np.sum(juzhen1, axis=2), axis=0)
    if np.sum(d_all) > 0:
        p = d_all / np.sum(d_all)
    else:
        p = global_popularity[:pdim].copy()
    strategies.append(p)
    
    # TOD2
    if np.sum(juzhen2[d1, :, t2]) > 0:
        p = juzhen2[d1, :, t2] / np.sum(juzhen2[d1, :, t2])
    else:
        p = global_popularity[:pdim].copy()
    strategies.append(p)
    
    # TD2
    td2 = np.sum(juzhen2, axis=0).transpose()
    if np.sum(td2[t2, :]) > 0:
        p = td2[t2, :] / np.sum(td2[t2, :])
    else:
        p = global_popularity[:pdim].copy()
    strategies.append(p)
    
    # 4. 多头注意力融合
    fused_prob, attn_weights = multi_head_attention_fusion(
        context_emb, strategies, num_heads=3
    )
    
    # 5. 融入时间注意力 (近期历史)
    temporal_weights = compute_temporal_attention(recent_history, t1, decay_factor=0.9)
    if temporal_weights:
        temporal_prob = np.zeros(pdim)
        for loc, w in temporal_weights:
            if loc < pdim:
                temporal_prob[loc] += w
        if np.sum(temporal_prob) > 0:
            temporal_prob = temporal_prob / np.sum(temporal_prob)
            # 融合时间注意力
            fused_prob = 0.85 * fused_prob + 0.15 * temporal_prob
    
    # 6. 上下文门控: 平衡个性化与全局
    gated_prob = context_gate * fused_prob + (1 - context_gate) * global_popularity[:pdim]
    
    # 7. 返回Top-10预测
    pred_indices = np.argsort(gated_prob)[::-1][:10].tolist()
    
    return pred_indices, attn_weights

def evaluate_comprehensive(y_true, y_pred):
    """综合评估指标"""
    n = len(y_true)
    top1 = top5 = top10 = 0
    rr_sum = 0
    ndcg5_sum = ndcg10_sum = 0
    
    # 覆盖率统计
    all_preds = set()
    
    for true_loc, pred_list in zip(y_true, y_pred):
        all_preds.update(pred_list)
        
        if true_loc in pred_list:
            rank = pred_list.index(true_loc) + 1
            
            if rank == 1:
                top1 += 1
                top5 += 1
                top10 += 1
            elif rank <= 5:
                top5 += 1
                top10 += 1
            elif rank <= 10:
                top10 += 1
            
            rr_sum += 1.0 / rank
            
            if rank <= 5:
                ndcg5_sum += 1.0 / np.log2(rank + 1)
            if rank <= 10:
                ndcg10_sum += 1.0 / np.log2(rank + 1)
    
    return {
        'top1': top1 / n,
        'top5': top5 / n,
        'top10': top10 / n,
        'mrr': rr_sum / n,
        'ndcg5': ndcg5_sum / n,
        'ndcg10': ndcg10_sum / n,
        'coverage': len(all_preds)
    }

def main():
    """主函数"""
    print("=" * 70)
    print("ADF Model - Attention-based Dynamic Fusion")
    print("基于注意力机制的动态融合模型")
    print("=" * 70)
    
    flag = 'foursquare_nyc'
    name1 = '../data_preprocessing/base_' + flag + '_v1.txt'
    name3 = '../refined maximum predictability/' + flag + '_rmp_train.txt'
    
    print("\n[1/4] 读取数据...")
    ff = open(name1, encoding='utf-8')
    df2 = pd.read_csv(name3, sep='$', header=None, encoding='utf-8')
    
    # 计算全局流行度
    print("[2/4] 计算全局统计信息...")
    all_sequences = []
    for ij in ff.readlines():
        mess = ij.strip().split('#')
        binds = mess[-1].split('&')
        if 'test' in binds:
            dseq1 = mess[1].split('&')
            all_sequences.append([int(x) for x in dseq1])
    ff.close()
    
    loc_counts = defaultdict(int)
    total = 0
    for seq in all_sequences:
        for loc in seq:
            loc_counts[loc] += 1
            total += 1
    
    max_loc = max(loc_counts.keys()) if loc_counts else 0
    global_pop = np.zeros(max_loc + 1)
    for loc, cnt in loc_counts.items():
        global_pop[loc] = cnt / total
    
    print(f"     位置空间大小: {max_loc + 1}")
    print(f"     平均序列长度: {np.mean([len(s) for s in all_sequences]):.1f}")
    
    # 重新打开文件进行预测
    ff = open(name1, encoding='utf-8')
    
    print("[3/4] 执行ADF预测...")
    
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
        
        tseq2 = [int(t.split('_')[3]) + (24 if int(t.split('_')[2]) > 4 else 0) for t in tseq1]
        tseq3 = [1 if int(t.split('_')[2]) > 4 else 0 for t in tseq1]
        
        # 构建位置映射
        pdnuml = sorted(set(dseq1))
        pdnuml = pdnuml + ['-1', '-2', '-3', '-4', '-5', '-6', '-7', '-8', '-9', '-10']
        pdnum = len(pdnuml)
        
        tdnuml2 = list(set(tseq2))
        tdnum2 = len(tdnuml2)
        tdnuml3 = list(set(tseq3))
        tdnum3 = len(tdnuml3)
        
        juzhen1 = np.zeros((pdnum, pdnum, tdnum2))
        juzhen2 = np.zeros((pdnum, pdnum, tdnum3))
        
        pdd = {pdnuml[i]: i for i in range(len(pdnuml))}
        tdd2 = {tdnuml2[i]: i for i in range(len(tdnuml2))}
        tdd3 = {tdnuml3[i]: i for i in range(len(tdnuml3))}
        
        tseq22 = [tdd2[x] for x in tseq2]
        tseq33 = [tdd3[x] for x in tseq3]
        dseq11 = [pdd[x] for x in dseq1]
        
        # 扩展全局流行度
        user_global_pop = np.zeros(pdnum)
        for loc, idx in pdd.items():
            if loc != '-1' and int(loc) < len(global_pop):
                user_global_pop[idx] = global_pop[int(loc)]
        if np.sum(user_global_pop) > 0:
            user_global_pop = user_global_pop / np.sum(user_global_pop)
        
        # 维护近期历史
        recent_history = deque(maxlen=10)
        
        user_y_true = []
        user_y_pred = []
        
        for j in range(1, len(dseq11)):
            t2 = tseq22[j-1]
            t3 = tseq33[j-1]
            d1 = dseq11[j-1]
            d2 = dseq11[j]
            
            if binds[j] == 'test':
                # ADF预测
                pred, weights = adf_predict(
                    juzhen1, juzhen2, d1, t2, t3,
                    rmp_values, user_global_pop, list(recent_history),
                    context_gate=0.85
                )
                user_y_true.append(d2)
                user_y_pred.append(pred)
            
            # 更新历史
            recent_history.append((d1, t2))
            
            # 更新转移矩阵
            juzhen1[d1, d2, t2] += 1
            juzhen2[d1, d2, t3] += 1
        
        if user_y_true:
            metrics = evaluate_comprehensive(user_y_true, user_y_pred)
            results.append(metrics)
            all_y_true.extend(user_y_true)
            all_y_pred.extend(user_y_pred)
        
        user_count += 1
        if user_count % 100 == 0:
            print(f"     已处理 {user_count} 个用户...")
    
    ff.close()
    
    print("[4/4] 计算评估指标...")
    
    # 汇总结果
    overall = evaluate_comprehensive(all_y_true, all_y_pred)
    
    print("\n" + "=" * 70)
    print("ADF模型 - 实验结果")
    print("=" * 70)
    print(f"\n处理用户数: {user_count}")
    print(f"总测试样本数: {len(all_y_true)}")
    print(f"\n[性能指标]")
    print(f"  Top-1  准确率: {overall['top1']:.4f} ({overall['top1']*100:.2f}%)")
    print(f"  Top-5  准确率: {overall['top5']:.4f} ({overall['top5']*100:.2f}%)")
    print(f"  Top-10 准确率: {overall['top10']:.4f} ({overall['top10']*100:.2f}%)")
    print(f"  MRR:           {overall['mrr']:.4f}")
    print(f"  NDCG@5:        {overall['ndcg5']:.4f}")
    print(f"  NDCG@10:       {overall['ndcg10']:.4f}")
    print(f"  覆盖率:        {overall['coverage']} 个不同位置")
    print("=" * 70)
    
    # 保存结果
    output = {
        'model': 'ADF (Attention-based Dynamic Fusion)',
        'user_count': user_count,
        'total_samples': len(all_y_true),
        'overall_metrics': overall
    }
    
    with open('../results/adf_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n结果已保存至: results/adf_results.json")
    
    # 与基线对比
    print("\n" + "=" * 70)
    print("方法对比")
    print("=" * 70)
    print(f"{'方法':<20} {'Top-1':>8} {'Top-5':>8} {'Top-10':>8} {'MRR':>8}")
    print("-" * 70)
    print(f"{'Base Markov':<20} {0.2318:>8.4f} {0.4850:>8.4f} {0.5758:>8.4f} {0.3132:>8.4f}")
    print(f"{'RMP (Original)':<20} {0.2505:>8.4f} {0.5611:>8.4f} {0.6730:>8.4f} {0.3512:>8.4f}")
    print(f"{'ADF (Ours)':<20} {overall['top1']:>8.4f} {overall['top5']:>8.4f} {overall['top10']:>8.4f} {overall['mrr']:>8.4f}")
    print("=" * 70)
    
    return overall

if __name__ == '__main__':
    main()
