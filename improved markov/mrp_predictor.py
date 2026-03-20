#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRP Predictor - 基于MRP可预测性指标的预测模型

使用MRP（而非RMP或ATDP）来指导预测策略

策略：
- 根据MRP值平滑调整策略权重
- 高MRP: 更多依赖个性化转移
- 低MRP: 更多依赖全局流行度
- 始终使用加权融合，避免硬切换

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

def mrp_guided_predict_smooth(juzhen1, juzhen2, d1, t1, t2, mrp, global_popularity):
    """
    基于MRP的平滑策略指导预测
    
    参数:
        juzhen1, juzhen2: 转移矩阵
        d1: 当前位置
        t1, t2: 时间特征
        mrp: MRP可预测性值 [0, 1]
        global_popularity: 全局流行度
    
    返回:
        predictions: Top-10预测列表
        strategy_weight: 使用的个性化权重
    """
    pdim = juzhen1.shape[1]
    
    # 平滑策略：根据MRP值调整个性化程度
    # MRP越高，个性化权重越大
    personalization_weight = mrp  # 直接使用MRP作为权重
    global_weight = 1 - mrp
    
    # 计算个性化概率 (TOD策略)
    if np.sum(juzhen1[d1, :, t1]) > 0:
        p_personal = juzhen1[d1, :, t1] / np.sum(juzhen1[d1, :, t1])
    else:
        p_personal = global_popularity[:pdim].copy()
    
    # 融合
    final_prob = personalization_weight * p_personal + global_weight * global_popularity[:pdim]
    
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
    print("MRP Predictor - 基于MRP的预测模型")
    print("=" * 60)
    
    func1 = lambda x: time1(x)
    func2 = lambda x: time2(x)
    
    flag = 'foursquare_nyc'
    name1 = '../data_preprocessing/base_' + flag + '_v1.txt'
    name_mrp = '../refined maximum predictability/' + flag + '_mrp_train.txt'
    
    print("\n[1/4] 读取数据...")
    ff = open(name1, encoding='utf-8')
    
    # 读取MRP值
    try:
        df_mrp = pd.read_csv(name_mrp, sep='$', header=None, encoding='utf-8')
        mrp_dict = dict(zip(df_mrp[0], df_mrp[2]))
        print(f"     成功读取MRP值，共 {len(mrp_dict)} 个用户")
    except Exception as e:
        print(f"     错误: 未找到MRP文件，请先运行 mrp_compute.py")
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
    
    print("[3/4] 执行MRP指导预测...")
    results = []
    all_y_true = []
    all_y_pred = []
    user_count = 0
    mrp_bins = {'low': 0, 'medium': 0, 'high': 0}
    
    for ij in ff.readlines():
        mess = ij.strip().split('#')
        binds = mess[-1].split('&')
        user = int(mess[0])
        
        if 'test' not in binds:
            continue
        
        # 获取MRP值
        mrp = mrp_dict.get(user, 0.5)
        
        # 统计MRP分布
        if mrp < 0.33:
            mrp_bins['low'] += 1
        elif mrp < 0.66:
            mrp_bins['medium'] += 1
        else:
            mrp_bins['high'] += 1
        
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
                # MRP指导预测
                pred, strategy_w = mrp_guided_predict_smooth(
                    juzhen1, juzhen2, d1, t2, t3, 
                    mrp, user_global
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
    print("MRP Predictor - 实验结果")
    print("=" * 60)
    print(f"处理用户数: {user_count}")
    print(f"总测试样本数: {len(all_y_true)}")
    print(f"\nMRP分布:")
    for bin_name, count in mrp_bins.items():
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
        'model': 'MRP Predictor',
        'user_count': user_count,
        'total_samples': len(all_y_true),
        'mrp_distribution': mrp_bins,
        'overall_metrics': overall
    }
    with open('../results/mrp_predictor_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n对比总结:")
    print("-" * 60)
    print(f"{'方法':<20} {'Top-1':>8} {'Top-5':>8} {'Top-10':>8} {'MRR':>8}")
    print("-" * 60)
    print(f"{'Base Markov':<20} {0.2318:>8.4f} {0.4850:>8.4f} {0.5758:>8.4f} {0.3132:>8.4f}")
    print(f"{'RMP (Original)':<20} {0.2505:>8.4f} {0.5611:>8.4f} {0.6730:>8.4f} {0.3512:>8.4f}")
    print(f"{'ATDP (Failed)':<20} {0.0904:>8.4f} {0.3750:>8.4f} {0.5451:>8.4f} {0.2113:>8.4f}")
    print(f"{'MRP (Ours)':<20} {overall['top1']:>8.4f} {overall['top5']:>8.4f} {overall['top10']:>8.4f} {overall['mrr']:>8.4f}")
    print("=" * 60)
    
    # 判断是否成功
    if overall['top10'] > 0.60:
        print("\n✅ MRP指标成功！超越ATDP，接近RMP水平")
    else:
        print("\n❌ MRP指标仍需改进")
    
    return overall

if __name__ == '__main__':
    main()
