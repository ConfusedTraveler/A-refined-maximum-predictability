#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATDP Predictor - 基于ATDP可预测性指标的预测模型

使用ATDP（而非原RMP）来指导预测策略选择

策略选择逻辑：
- ATDP > 0.8: 高可预测性，使用个性化模型（TOD）
- 0.5 < ATDP <= 0.8: 中等可预测性，加权融合
- ATDP <= 0.5: 低可预测性，依赖全局趋势

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
    if ww > 4:
        nhh = hh + 24
    else:
        nhh = hh
    return nhh

def time2(x):
    """提取工作日/周末特征"""
    mm = x.split('_')
    ww = int(mm[2])
    return 1 if ww > 4 else 0

def compute_temporal_weights_local(sequence_length, lambda_decay=0.3):
    """计算局部时间衰减权重"""
    weights = np.zeros(sequence_length)
    for t in range(sequence_length):
        weights[t] = np.exp(-lambda_decay * np.sqrt(t))
    return weights / np.sum(weights) if np.sum(weights) > 0 else weights

def atdp_guided_predict(juzhen1, juzhen2, d1, t1, t2, atdp, global_popularity, 
                       history_weights=None):
    """
    基于ATDP的策略指导预测
    
    参数:
        juzhen1, juzhen2: 转移矩阵
        d1: 当前位置
        t1, t2: 时间特征
        atdp: ATDP可预测性值
        global_popularity: 全局流行度
        history_weights: 历史权重（用于加权）
    
    返回:
        predictions: Top-10预测列表
        strategy_used: 使用的策略
    """
    pdim = juzhen1.shape[1]
    
    # 根据ATDP值选择策略
    if atdp > 0.8:
        # 高可预测性: 主要依赖个性化TOD
        strategy_used = "high_personalization"
        
        if np.sum(juzhen1[d1, :, t1]) > 0:
            p_tod = juzhen1[d1, :, t1] / np.sum(juzhen1[d1, :, t1])
        else:
            p_tod = global_popularity[:pdim].copy()
        
        # 稍微融合一点全局信息
        final_prob = 0.9 * p_tod + 0.1 * global_popularity[:pdim]
        
    elif atdp > 0.5:
        # 中等可预测性: 加权融合
        strategy_used = "adaptive_fusion"
        
        # 计算各策略概率
        strategies = []
        
        # TOD
        if np.sum(juzhen1[d1, :, t1]) > 0:
            strategies.append(juzhen1[d1, :, t1] / np.sum(juzhen1[d1, :, t1]))
        else:
            strategies.append(global_popularity[:pdim].copy())
        
        # OD
        od = np.sum(juzhen1, axis=2)
        if np.sum(od[d1, :]) > 0:
            strategies.append(od[d1, :] / np.sum(od[d1, :]))
        else:
            strategies.append(global_popularity[:pdim].copy())
        
        # D (全局)
        d_all = np.sum(np.sum(juzhen1, axis=2), axis=0)
        if np.sum(d_all) > 0:
            strategies.append(d_all / np.sum(d_all))
        else:
            strategies.append(global_popularity[:pdim].copy())
        
        # 根据ATDP调整权重
        # ATDP越高，个性化权重越大
        w_personal = atdp
        w_global = 1 - atdp
        
        final_prob = w_personal * 0.7 * strategies[0] + \
                     w_personal * 0.3 * strategies[1] + \
                     w_global * strategies[2]
        
    else:
        # 低可预测性: 主要依赖全局趋势
        strategy_used = "global_trend"
        
        # 使用全局流行度为主
        final_prob = 0.7 * global_popularity[:pdim] + 0.3 * np.ones(pdim) / pdim
    
    # 如果时间权重提供，应用时间衰减
    if history_weights is not None and len(history_weights) > 0:
        # 创建基于历史的偏置
        history_bias = np.zeros(pdim)
        for loc, w in history_weights:
            if loc < pdim:
                history_bias[loc] += w
        if np.sum(history_bias) > 0:
            history_bias = history_bias / np.sum(history_bias)
            final_prob = 0.85 * final_prob + 0.15 * history_bias
    
    # 返回Top-10
    pred_indices = np.argsort(final_prob)[::-1][:10].tolist()
    return pred_indices, strategy_used

def evaluate_metrics(y_true, y_pred):
    """综合评估指标"""
    n = len(y_true)
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
    print("ATDP Predictor - 基于ATDP的预测模型")
    print("=" * 60)
    
    func1 = lambda x: time1(x)
    func2 = lambda x: time2(x)
    
    flag = 'foursquare_nyc'
    name1 = '../data_preprocessing/base_' + flag + '_v1.txt'
    name_atdp = '../refined maximum predictability/' + flag + '_atdp_train.txt'
    
    print("\n[1/4] 读取数据...")
    ff = open(name1, encoding='utf-8')
    
    # 读取ATDP值
    try:
        df_atdp = pd.read_csv(name_atdp, sep='$', header=None, encoding='utf-8')
        atdp_dict = dict(zip(df_atdp[0], df_atdp[2]))
        print(f"     成功读取ATDP值，共 {len(atdp_dict)} 个用户")
    except:
        print("     错误: 未找到ATDP文件，请先运行 atdp_compute.py")
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
    
    print("[3/4] 执行ATDP指导预测...")
    results = []
    all_y_true = []
    all_y_pred = []
    user_count = 0
    strategy_counts = {'high_personalization': 0, 'adaptive_fusion': 0, 'global_trend': 0}
    
    for ij in ff.readlines():
        mess = ij.strip().split('#')
        binds = mess[-1].split('&')
        user = int(mess[0])
        
        if 'test' not in binds:
            continue
        
        # 获取ATDP值
        atdp = atdp_dict.get(user, 0.5)
        
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
        recent_history = []
        
        for j in range(1, len(dseq11)):
            t2 = tseq22[j-1]
            t3 = tseq33[j-1]
            d1 = dseq11[j-1]
            d2 = dseq11[j]
            
            if binds[j] == 'test':
                # 计算局部时间权重
                local_weights = compute_temporal_weights_local(len(recent_history), lambda_decay=0.3)
                history_w = [(recent_history[k], local_weights[k]) for k in range(len(recent_history))] if len(recent_history) > 0 else []
                
                # ATDP指导预测
                pred, strategy = atdp_guided_predict(
                    juzhen1, juzhen2, d1, t2, t3, 
                    atdp, user_global, history_w
                )
                strategy_counts[strategy] += 1
                
                user_y_true.append(d2)
                user_y_pred.append(pred)
            
            recent_history.append(d1)
            
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
    print("ATDP Predictor - 实验结果")
    print("=" * 60)
    print(f"处理用户数: {user_count}")
    print(f"总测试样本数: {len(all_y_true)}")
    print(f"\n策略使用统计:")
    total_strat = sum(strategy_counts.values())
    for strat, count in strategy_counts.items():
        print(f"  {strat}: {count} ({count/total_strat*100:.1f}%)")
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
        'model': 'ATDP Predictor',
        'user_count': user_count,
        'total_samples': len(all_y_true),
        'strategy_distribution': strategy_counts,
        'overall_metrics': overall
    }
    with open('../results/atdp_predictor_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n对比总结:")
    print("-" * 60)
    print(f"{'方法':<20} {'Top-1':>8} {'Top-5':>8} {'Top-10':>8} {'MRR':>8}")
    print("-" * 60)
    print(f"{'Base Markov':<20} {0.2318:>8.4f} {0.4850:>8.4f} {0.5758:>8.4f} {0.3132:>8.4f}")
    print(f"{'RMP (Original)':<20} {0.2505:>8.4f} {0.5611:>8.4f} {0.6730:>8.4f} {0.3512:>8.4f}")
    print(f"{'ATDP (Ours)':<20} {overall['top1']:>8.4f} {overall['top5']:>8.4f} {overall['top10']:>8.4f} {overall['mrr']:>8.4f}")
    print("=" * 60)
    
    return overall

if __name__ == '__main__':
    main()
