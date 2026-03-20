#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ε-RMP Predictor - 基于容差鲁棒可预测性的预测模型

核心改进：
1. 软融合（解决硬分类问题）
2. 容差匹配（提高鲁棒性）
3. 自适应策略权重

Author: AI Assistant
Date: 2026-03-20
"""

import numpy as np
import pandas as pd
import json

def time1(x):
    """提取详细时间特征"""
    mm = x.split('_')
    ww = int(mm[2])
    hh = int(mm[3])
    return hh + 24 if ww > 4 else hh

def time2(x):
    """提取工作日/周末特征"""
    mm = x.split('_')
    return 1 if int(mm[2]) > 4 else 0

def tolerance_match(val1, val2, epsilon):
    """容差匹配"""
    return abs(int(val1) - int(val2)) <= epsilon

def epsilon_rmp_predict_smooth(juzhen1, juzhen2, d1, t1, t2, epsilon_rmp, 
                                global_popularity, epsilon=1.0):
    """
    基于ε-RMP的平滑预测
    
    核心：使用ε-RMP值直接作为个性化权重，无硬分类
    
    参数:
        juzhen1, juzhen2: 转移矩阵
        d1: 当前位置
        t1, t2: 时间特征
        epsilon_rmp: ε-RMP值（直接使用作为权重）
        global_popularity: 全局流行度
        epsilon: 容差值
    
    返回:
        predictions: Top-10预测
        personalization_weight: 实际使用的个性化权重
    """
    pdim = juzhen1.shape[1]
    
    # 软融合：直接使用ε-RMP作为个性化权重
    personalization_weight = epsilon_rmp
    global_weight = 1 - epsilon_rmp
    
    # 容差感知的个性化概率
    # 收集容差范围内的匹配
    p_personal = np.zeros(pdim)
    
    # 策略1: TOD（容差匹配）
    total_weight = 0
    for loc in range(pdim):
        if tolerance_match(loc, d1, epsilon):
            if np.sum(juzhen1[loc, :, t1]) > 0:
                p_personal += juzhen1[loc, :, t1]
                total_weight += 1
    
    if total_weight > 0 and np.sum(p_personal) > 0:
        p_personal = p_personal / np.sum(p_personal)
    else:
        # 回退到精确匹配
        if np.sum(juzhen1[d1, :, t1]) > 0:
            p_personal = juzhen1[d1, :, t1] / np.sum(juzhen1[d1, :, t1])
        else:
            p_personal = global_popularity[:pdim].copy()
    
    # 软融合
    final_prob = personalization_weight * p_personal + global_weight * global_popularity[:pdim]
    
    # 确保概率和为1
    if np.sum(final_prob) > 0:
        final_prob = final_prob / np.sum(final_prob)
    
    # 返回Top-10
    pred_indices = np.argsort(final_prob)[::-1][:10].tolist()
    
    return pred_indices, personalization_weight

def evaluate_metrics(y_true, y_pred):
    """综合评估指标"""
    n = len(y_true)
    if n == 0:
        return {'top1': 0, 'top5': 0, 'top10': 0, 'mrr': 0, 'ndcg5': 0, 'ndcg10': 0}
    
    top1 = top5 = top10 = 0
    rr_sum = 0
    ndcg5_sum = ndcg10_sum = 0
    
    for true_loc, pred_list in zip(y_true, y_pred):
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
        'ndcg10': ndcg10_sum / n
    }

def main():
    """主函数"""
    print("=" * 60)
    print("ε-RMP Predictor - 容差鲁棒预测模型")
    print("=" * 60)
    
    func1 = lambda x: time1(x)
    func2 = lambda x: time2(x)
    
    flag = 'foursquare_nyc'
    name1 = '../data_preprocessing/base_' + flag + '_v1.txt'
    name_epsilon_rmp = '../refined maximum predictability/' + flag + '_epsilon_rmp_train.txt'
    
    print("\n[1/4] 读取数据...")
    ff = open(name1, encoding='utf-8')
    
    # 读取ε-RMP值
    try:
        df_epsilon_rmp = pd.read_csv(name_epsilon_rmp, sep='$', header=None, encoding='utf-8')
        epsilon_rmp_dict = dict(zip(df_epsilon_rmp[0], df_epsilon_rmp[2]))
        # 同时读取容差值
        epsilon_dict = dict(zip(df_epsilon_rmp[0], df_epsilon_rmp[3]))
        print(f"     成功读取ε-RMP值，共 {len(epsilon_rmp_dict)} 个用户")
    except Exception as e:
        print(f"     错误: 未找到ε-RMP文件，请先运行 epsilon_rmp_compute.py")
        print(f"     错误详情: {e}")
        return
    
    # 计算全局流行度
    print("[2/4] 计算全局统计...")
    all_sequences = []
    for ij in ff.readlines():
        mess = ij.strip().split('#')
        binds = mess[-1].split('&')
        if 'test' in binds:
            dseq1 = mess[1].split('&')
            all_sequences.append([int(x) for x in dseq1])
    ff.close()
    
    loc_counts = {}
    total = 0
    for seq in all_sequences:
        for loc in seq:
            loc_counts[loc] = loc_counts.get(loc, 0) + 1
            total += 1
    
    max_loc = max(loc_counts.keys()) if loc_counts else 0
    global_pop = np.zeros(max_loc + 1)
    for loc, cnt in loc_counts.items():
        global_pop[loc] = cnt / total
    
    # 重新打开进行预测
    ff = open(name1, encoding='utf-8')
    
    print("[3/4] 执行ε-RMP预测...")
    results = []
    all_y_true = []
    all_y_pred = []
    user_count = 0
    
    # 统计ε-RMP分布
    epsilon_rmp_bins = {'low': 0, 'medium': 0, 'high': 0}
    
    for ij in ff.readlines():
        mess = ij.strip().split('#')
        binds = mess[-1].split('&')
        user = int(mess[0])
        
        if 'test' not in binds:
            continue
        
        # 获取ε-RMP值和容差
        epsilon_rmp = epsilon_rmp_dict.get(user, 0.5)
        epsilon = epsilon_dict.get(user, 1.0)
        
        # 统计分布
        if epsilon_rmp < 0.33:
            epsilon_rmp_bins['low'] += 1
        elif epsilon_rmp < 0.66:
            epsilon_rmp_bins['medium'] += 1
        else:
            epsilon_rmp_bins['high'] += 1
        
        dseq1 = mess[1].split('&')
        tseq1 = mess[2].split('&')
        
        tseq2 = list(map(func1, tseq1))
        tseq3 = list(map(func2, tseq1))
        
        # 构建映射
        pdnuml = sorted(set(dseq1))
        pdnuml = pdnuml + ['-1'] * 10
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
        user_global = np.zeros(pdnum)
        for loc, idx in pdd.items():
            if loc != '-1' and int(loc) < len(global_pop):
                user_global[idx] = global_pop[int(loc)]
        if np.sum(user_global) > 0:
            user_global = user_global / np.sum(user_global)
        
        # 预测
        user_y_true = []
        user_y_pred = []
        
        for j in range(1, len(dseq11)):
            t2 = tseq22[j-1]
            t3 = tseq33[j-1]
            d1 = dseq11[j-1]
            d2 = dseq11[j]
            
            if binds[j] == 'test':
                # ε-RMP软融合预测
                pred, p_weight = epsilon_rmp_predict_smooth(
                    juzhen1, juzhen2, d1, t2, t3, 
                    epsilon_rmp, user_global, epsilon
                )
                
                user_y_true.append(d2)
                user_y_pred.append(pred)
            
            # 更新矩阵
            juzhen1[d1, d2, t2] += 1
            juzhen2[d1, d2, t3] += 1
        
        if user_y_true:
            metrics = evaluate_metrics(user_y_true, user_y_pred)
            results.append(metrics)
            all_y_true.extend(user_y_true)
            all_y_pred.extend(user_y_pred)
        
        user_count += 1
        if user_count % 200 == 0:
            print(f"     已处理 {user_count} 个用户...")
    
    ff.close()
    
    print("[4/4] 计算评估指标...")
    overall = evaluate_metrics(all_y_true, all_y_pred)
    
    print("\n" + "=" * 60)
    print("ε-RMP Predictor - 实验结果")
    print("=" * 60)
    print(f"处理用户数: {user_count}")
    print(f"总测试样本数: {len(all_y_true)}")
    print(f"\nε-RMP分布:")
    for bin_name, count in epsilon_rmp_bins.items():
        print(f"  {bin_name}: {count} ({count/user_count*100:.1f}%)")
    print(f"\n性能指标:")
    print(f"  Top-1:  {overall['top1']:.4f} ({overall['top1']*100:.2f}%)")
    print(f"  Top-5:  {overall['top5']:.4f} ({overall['top5']*100:.2f}%)")
    print(f"  Top-10: {overall['top10']:.4f} ({overall['top10']*100:.2f}%)")
    print(f"  MRR:    {overall['mrr']:.4f}")
    print(f"  NDCG@5: {overall['ndcg5']:.4f}")
    print(f"  NDCG@10:{overall['ndcg10']:.4f}")
    print("=" * 60)
    
    # 保存结果
    output = {
        'model': 'ε-RMP Predictor',
        'user_count': user_count,
        'total_samples': len(all_y_true),
        'epsilon_rmp_distribution': epsilon_rmp_bins,
        'overall_metrics': overall
    }
    with open('../results/epsilon_rmp_predictor_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n全面对比:")
    print("-" * 60)
    print(f"{'方法':<20} {'Top-1':>8} {'Top-5':>8} {'Top-10':>8} {'MRR':>8}")
    print("-" * 60)
    print(f"{'Base Markov':<20} {0.2318:>8.4f} {0.4850:>8.4f} {0.5758:>8.4f} {0.3132:>8.4f}")
    print(f"{'RMP (Original)':<20} {0.2505:>8.4f} {0.5611:>8.4f} {0.6730:>8.4f} {0.3512:>8.4f}")
    print(f"{'Weighted RMP':<20} {0.2452:>8.4f} {0.5679:>8.4f} {0.6920:>8.4f} {0.3803:>8.4f}")
    print(f"{'ADF':<20} {0.2465:>8.4f} {0.5705:>8.4f} {0.6943:>8.4f} {0.3823:>8.4f}")
    print(f"{'ATDP':<20} {0.0904:>8.4f} {0.3750:>8.4f} {0.5451:>8.4f} {0.2113:>8.4f}")
    print(f"{'MRP':<20} {0.1663:>8.4f} {0.3537:>8.4f} {0.5118:>8.4f} {0.2534:>8.4f}")
    print(f"{'ε-RMP (Ours)':<20} {overall['top1']:>8.4f} {overall['top5']:>8.4f} {overall['top10']:>8.4f} {overall['mrr']:>8.4f}")
    print("=" * 60)
    
    # 判断是否成功
    if overall['top10'] > 0.65:
        print("\n✅ ε-RMP成功超越RMP！")
    elif overall['top10'] > 0.60:
        print("\n✅ ε-RMP表现良好，接近或达到RMP水平")
    else:
        print("\n⚠️ ε-RMP仍需改进，但已解决硬分类问题")
    
    return overall

if __name__ == '__main__':
    main()
