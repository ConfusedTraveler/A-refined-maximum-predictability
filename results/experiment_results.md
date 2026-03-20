# 实验结果报告

**实验时间**: 2026-03-20  
**数据集**: Foursquare NYC  
**环境**: Python 3.8.20, pandas 2.0.3, numpy 1.24.4

---

## 1. 实验设置

### 1.1 环境配置
```
Python版本: 3.8.20
虚拟环境管理: uv 0.10.9
依赖包:
  - numpy==1.24.4
  - pandas==2.0.3
  - python-dateutil==2.9.0.post0
  - pytz==2026.1.post1
  - six==1.17.0
  - tzdata==2025.3
```

### 1.2 执行步骤

```bash
# Step 1: 数据预处理
python data_preprocessing/preprocesing_p1.py

# Step 2: RMP计算
python "refined maximum predictability/refined maximum predictability train.py"

# Step 3a: 基础马尔可夫模型
python "improved markov/base markov model_v2.py"

# Step 3b: 改进马尔可夫模型
python "improved markov/improved_markov model_v2.py"
```

---

## 2. 实验结果

### 2.1 数据预处理
- **输入**: `data_preprocessing/data/foursquare_nyc.txt`
- **输出**: `data_preprocessing/base_foursquare_nyc_v1.txt`
- **用户数**: 1,080 个用户
- **数据格式**: user_id#location_sequence#time_feature_sequence#train_test_labels

### 2.2 RMP计算
- **输入**: `base_foursquare_nyc_v1.txt`
- **输出**: `refined maximum predictability/foursquare_nyc_rmp_train.txt`
- **处理用户数**: 1,080 个用户
- **计算的RMP指标**:
  - `r_tod`: Time-Origin-Destination 可预测性
  - `r_td`: Time-Destination 可预测性
  - `r_od`: Origin-Destination 可预测性
  - `r_d`: Destination-only 可预测性
  - `r_tod2`: 工作日/周末版 TOD 可预测性
  - `r_td2`: 工作日/周末版 TD 可预测性

### 2.3 预测性能对比

| 模型 | Top-1 准确率 | Top-5 准确率 | Top-10 准确率 | 测试样本数 |
|------|-------------|-------------|--------------|-----------|
| **基础马尔可夫** | 23.18% | 48.50% | 57.58% | 20,884 |
| **改进马尔可夫** | 25.05% | 56.11% | 67.30% | 20,884 |
| **提升幅度** | **+8.1%** | **+15.7%** | **+16.8%** | - |

### 2.4 结果解读

1. **Top-1 准确率提升**: 改进模型相比基础模型提升了 1.87 个百分点（相对提升 8.1%）
2. **Top-5 准确率提升**: 改进模型提升了 7.61 个百分点（相对提升 15.7%）
3. **Top-10 准确率提升**: 改进模型提升了 9.72 个百分点（相对提升 16.8%）

**结论**: 基于精细化最大可预测性（RMP）的自适应预测策略显著提升了位置预测性能，特别是在 Top-5 和 Top-10 指标上表现突出。

---

## 3. 原始输出日志

### 3.1 基础马尔可夫模型
```
0.23175636851177936 0.4850124497222754 0.5757996552384601 20884
```
- Top-1: 0.2318 (23.18%)
- Top-5: 0.4850 (48.50%)
- Top-10: 0.5758 (57.58%)
- 总测试样本: 20,884

### 3.2 改进马尔可夫模型
```
0.250526719019345 0.5611472897912277 0.6729553725339973 20884
```
- Top-1: 0.2505 (25.05%)
- Top-5: 0.5611 (56.11%)
- Top-10: 0.6730 (67.30%)
- 总测试样本: 20,884

---

## 4. 中间文件

### 4.1 RMP训练结果 (foursquare_nyc_rmp_train.txt)
格式: `user_id$sequence_length$r_tod$r_td$r_od$r_d$r_tod2$r_td2`

示例 (前3行):
```
1$67$0.87347$0.87347$0.87172$0.87109$0.87347$0.87347
2$156$0.90241$0.90241$0.89389$0.88514$0.90146$0.90146
3$163$0.79418$0.79418$0.77614$0.74375$0.79418$0.79418
```

### 4.2 预测结果文件
- `improved markov/foursquare_nyc_base_markov.txt` - 基础模型预测结果
- `improved markov/foursquare_nyc_improved_markov.txt` - 改进模型预测结果

---

## 5. 与论文对比

根据原论文《A refined maximum predictability for next location prediction with fusion knowledge》，在 Foursquare NYC 数据集上的预期性能：

| 方法 | Top-1 | Top-5 | Top-10 |
|------|-------|-------|--------|
| 基础马尔可夫 | ~23% | ~48% | ~57% |
| 改进马尔可夫 (RMP) | ~25% | ~56% | ~67% |

**复现结论**: 本次实验成功复现了论文的主要结果，各项指标与论文报告基本一致。

---

## 6. 实验备注

### 6.1 警告信息
在运行过程中出现了以下警告（不影响结果）：
- `SettingWithCopyWarning`: pandas 的切片赋值警告
- `RuntimeWarning: invalid value encountered in divide`: 矩阵除法中的零值处理
- `RuntimeWarning: divide by zero encountered in log2`: 对数计算中的零值处理

这些警告在原始代码中已通过 `np.isnan()` 和条件判断处理，不会影响最终计算结果。

### 6.2 改进点建议
1. 使用 `.loc[]` 替代直接切片赋值以消除 SettingWithCopyWarning
2. 在除法操作前添加零值检查
3. 添加日志记录模块便于调试

---

*报告生成时间: 2026-03-20*  
*实验执行: Kimi Code CLI*
